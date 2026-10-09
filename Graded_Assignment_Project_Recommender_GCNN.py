"""Recommender demo: matrix factorisation (BPR) vs. a graph model (LightGCN-style).

Runs on NumPy only, so it deploys on Streamlit Community Cloud without PyTorch.
Upload Rec_sys_data.xlsx (sheets: order, customer, product) or use the built-in sample.

Method
------
1. Implicit feedback: a customer "interacts" with a product if they bought it.
2. BPR-MF (Rendle et al., 2009): learn user/item embeddings so that, for a user u,
   a bought item i scores above a random unbought item j:  maximise ln σ(e_u·e_i − e_u·e_j).
3. Graph propagation (He et al., 2020, LightGCN): average embeddings over the
   normalised user–item graph  E^(k+1) = D^-1/2 A D^-1/2 E^(k),  final E = mean_k E^(k).
   Neighbours' tastes flow into each user, which helps sparse users.
4. Evaluation: hold out one purchase per user and report Hit-Rate@K and NDCG@K.
"""
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Recommender: BPR-MF vs. Graph (LightGCN)", page_icon="🧠", layout="wide")
st.title("🧠 Recommender System: Matrix Factorisation vs. Graph Neural Propagation")
st.caption("BPR-MF (Rendle et al., 2009) vs. LightGCN-style propagation (He et al., 2020). NumPy implementation.")


# ---------- data ----------
@st.cache_data
def sample_data(n_users=300, n_items=120, seed=7):
    """Synthetic shop: customers belong to taste segments that prefer product categories."""
    rng = np.random.default_rng(seed)
    cats = ["Beverages", "Snacks", "Dairy", "Bakery", "Household", "Personal Care"]
    brands = ["Aurora", "Nimbus", "Zenith", "Orchid", "Atlas"]
    product = pd.DataFrame({
        "StockCode": [f"P{i:04d}" for i in range(n_items)],
        "Category": rng.choice(cats, n_items),
        "Brand": rng.choice(brands, n_items),
        "Unit Price": rng.uniform(0.5, 25, n_items).round(2),
    })
    product["Product Name"] = product["Brand"] + " " + product["Category"] + " #" + product.index.astype(str)
    customer = pd.DataFrame({"CustomerID": np.arange(1000, 1000 + n_users),
                             "Segment": rng.integers(0, len(cats), n_users)})
    rows = []
    for cid, seg in zip(customer.CustomerID, customer.Segment):
        pref = np.where(product.Category == cats[seg], 15.0, 1.0)
        pref /= pref.sum()
        for code in rng.choice(product.StockCode, size=rng.integers(5, 18), replace=False, p=pref):
            rows.append((cid, code, int(rng.integers(1, 6))))
    order = pd.DataFrame(rows, columns=["CustomerID", "StockCode", "Quantity"])
    return order, customer, product


def load(upload):
    if upload is None:
        return sample_data()
    xl = pd.read_excel(upload, sheet_name=None)
    missing = {"order", "customer", "product"} - set(xl)
    if missing:
        st.error(f"The workbook needs sheets named order, customer and product. Missing: {', '.join(sorted(missing))}.")
        st.stop()
    return xl["order"], xl["customer"], xl["product"]


# ---------- models ----------
def split_leave_one_out(pairs, n_users, rng):
    """Hold out one item per user (users with ≥2 items) for evaluation."""
    test = {}
    keep = np.ones(len(pairs), bool)
    for u in range(n_users):
        idx = np.flatnonzero(pairs[:, 0] == u)
        if len(idx) >= 2:
            j = rng.choice(idx)
            test[u] = pairs[j, 1]
            keep[j] = False
    return pairs[keep], test


@st.cache_data(show_spinner=False)
def train_bpr(train, n_users, n_items, dim=16, epochs=60, lr=0.05, reg=1e-2, seed=0):
    rng = np.random.default_rng(seed)
    U = rng.normal(0, 0.1, (n_users, dim))
    V = rng.normal(0, 0.1, (n_items, dim))
    seen = [set() for _ in range(n_users)]
    for u, i in train:
        seen[u].add(i)
    losses = []
    for _ in range(epochs):
        order = rng.permutation(len(train))
        u, i = train[order, 0], train[order, 1]
        j = rng.integers(0, n_items, len(order))
        for t in range(len(order)):                       # resample negatives the user already bought
            while j[t] in seen[u[t]]:
                j[t] = rng.integers(0, n_items)
        total = 0.0
        for b in range(0, len(order), 256):               # mini-batch SGD
            ub, ib, jb = u[b:b + 256], i[b:b + 256], j[b:b + 256]
            x = np.sum(U[ub] * (V[ib] - V[jb]), axis=1)
            g = 1 / (1 + np.exp(x))                       # d/dx of -ln σ(x) is -(1-σ(x))
            total += np.sum(np.logaddexp(0, -x))
            du = g[:, None] * (V[ib] - V[jb]) - reg * U[ub]
            di = g[:, None] * U[ub] - reg * V[ib]
            dj = -g[:, None] * U[ub] - reg * V[jb]
            np.add.at(U, ub, lr * du)
            np.add.at(V, ib, lr * di)
            np.add.at(V, jb, lr * dj)
        losses.append(total / len(order))
    return U, V, losses


def propagate(U, V, train, n_users, n_items, layers=2):
    """LightGCN propagation on the symmetric-normalised bipartite graph."""
    du = np.bincount(train[:, 0], minlength=n_users).astype(float)
    di = np.bincount(train[:, 1], minlength=n_items).astype(float)
    w = 1 / np.sqrt(np.maximum(du[train[:, 0]], 1) * np.maximum(di[train[:, 1]], 1))
    Eu, Ei = [U], [V]
    for _ in range(layers):
        nu = np.zeros_like(U)
        ni = np.zeros_like(V)
        np.add.at(nu, train[:, 0], w[:, None] * Ei[-1][train[:, 1]])
        np.add.at(ni, train[:, 1], w[:, None] * Eu[-1][train[:, 0]])
        Eu.append(nu)
        Ei.append(ni)
    return np.mean(Eu, axis=0), np.mean(Ei, axis=0)


def evaluate(U, V, train, test, k):
    seen = {}
    for u, i in train:
        seen.setdefault(u, set()).add(i)
    hits, ndcg = [], []
    for u, held in test.items():
        s = V @ U[u]
        s[list(seen.get(u, ()))] = -np.inf
        top = np.argpartition(-s, k)[:k]
        top = top[np.argsort(-s[top])]
        pos = np.flatnonzero(top == held)
        hits.append(len(pos) > 0)
        ndcg.append(1 / np.log2(pos[0] + 2) if len(pos) else 0.0)
    return float(np.mean(hits)), float(np.mean(ndcg))


# ---------- UI ----------
with st.sidebar:
    st.header("Data")
    upload = st.file_uploader("Rec_sys_data.xlsx (optional)", type="xlsx")
    st.caption("Without a file, a synthetic shop with taste segments is used.")
    st.header("Model")
    dim = st.slider("Embedding size", 8, 64, 16, 8)
    epochs = st.slider("BPR epochs", 5, 100, 60, 5)
    layers = st.slider("Graph layers (LightGCN)", 1, 4, 2)
    k = st.slider("Top-K", 5, 20, 10)

df_order, df_customer, df_product = load(upload)
df = df_order.merge(df_product, on="StockCode", how="left").dropna(subset=["Category"])
grouped = df.groupby(["CustomerID", "StockCode"])["Quantity"].sum().reset_index()
users = pd.Index(grouped.CustomerID.astype(str).unique())
items = pd.Index(grouped.StockCode.astype(str).unique())
pairs = np.column_stack([users.get_indexer(grouped.CustomerID.astype(str)), items.get_indexer(grouped.StockCode.astype(str))])
n_users, n_items = len(users), len(items)

c1, c2, c3 = st.columns(3)
c1.metric("Customers", f"{n_users:,}")
c2.metric("Products", f"{n_items:,}")
c3.metric("Density", f"{len(pairs) / (n_users * n_items):.2%}")

rng = np.random.default_rng(42)
train, test = split_leave_one_out(pairs, n_users, rng)
with st.spinner("Training BPR matrix factorisation…"):
    U, V, losses = train_bpr(train, n_users, n_items, dim=dim, epochs=epochs)
Ug, Vg = propagate(U, V, train, n_users, n_items, layers=layers)

hr_mf, nd_mf = evaluate(U, V, train, test, k)
hr_g, nd_g = evaluate(Ug, Vg, train, test, k)
popularity = np.bincount(train[:, 1], minlength=n_items).astype(float)[:, None]
hr_p, nd_p = evaluate(np.ones((n_users, 1)), popularity, train, test, k)
k1, k2 = st.columns(2)
with k1:
    st.subheader("Held-out evaluation (leave-one-out)")
    st.dataframe(pd.DataFrame({"Model": ["Popularity baseline", "BPR-MF", f"Graph (LightGCN, {layers} layers)"],
                               f"Hit-Rate@{k}": [hr_p, hr_mf, hr_g], f"NDCG@{k}": [nd_p, nd_mf, nd_g]})
                 .style.format({f"Hit-Rate@{k}": "{:.3f}", f"NDCG@{k}": "{:.3f}"}), hide_index=True)
    st.caption(f"{len(test)} customers each have one purchase hidden; we check whether it appears in their top {k}.")
with k2:
    st.subheader("BPR training loss")
    st.line_chart(pd.DataFrame({"loss": losses}))

st.subheader("Recommendations")
model = st.radio("Model", ["Graph (LightGCN)", "BPR-MF"], horizontal=True)
cid = st.selectbox("Customer", users)
u = users.get_loc(cid)
Eu, Ei = (Ug, Vg) if model.startswith("Graph") else (U, V)
scores = Ei @ Eu[u]
bought = pairs[pairs[:, 0] == u, 1]
scores[bought] = -np.inf
top = np.argsort(-scores)[:k]
cols = [c for c in ["StockCode", "Product Name", "Category", "Brand", "Unit Price"] if c in df_product]
prod = df_product.assign(StockCode=df_product.StockCode.astype(str)).set_index("StockCode")
rec = prod.loc[items[top]].rename_axis("StockCode").reset_index()[cols]
rec.insert(0, "Score", scores[top].round(3))
hist = prod.loc[items[bought]].rename_axis("StockCode").reset_index()[cols]
a, b = st.columns(2)
a.markdown("**Already bought**")
a.dataframe(hist, hide_index=True)
b.markdown(f"**Top {k} recommendations**")
b.dataframe(rec, hide_index=True)

with st.expander("How it works"):
    st.markdown(__doc__)
    st.markdown("- Rendle et al. (2009) *BPR: Bayesian Personalized Ranking from Implicit Feedback*, UAI. "
                "[arXiv:1205.2618](https://arxiv.org/abs/1205.2618)\n"
                "- He et al. (2020) *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation*, SIGIR. "
                "[arXiv:2002.02126](https://arxiv.org/abs/2002.02126)")
