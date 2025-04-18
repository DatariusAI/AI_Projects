# app.py
import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
import numpy as np

st.title("🧠 Recommender System with GCNN vs. NeuMF")

# Upload Excel data
uploaded_file = st.file_uploader("Upload Rec_sys_data.xlsx", type="xlsx")

if uploaded_file:
    st.success("✅ File uploaded successfully!")
    
    # Load sheets
    df_order = pd.read_excel(uploaded_file, sheet_name='order')
    df_customer = pd.read_excel(uploaded_file, sheet_name='customer')
    df_product = pd.read_excel(uploaded_file, sheet_name='product')

    # Merge all
    df_order_customer = pd.merge(df_order, df_customer, on='CustomerID', how='left')
    df_full = pd.merge(df_order_customer, df_product, on='StockCode', how='left')

    # Build interaction matrix
    df = df_full.dropna(subset=['Product Name', 'Category'])
    df_grouped = df.groupby(['CustomerID', 'StockCode'])['Quantity'].sum().reset_index()

    # Encode
    customer_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    df_grouped['customer_idx'] = customer_encoder.fit_transform(df_grouped['CustomerID'].astype(str))
    df_grouped['item_idx'] = item_encoder.fit_transform(df_grouped['StockCode'].astype(str))

    num_users = df_grouped['customer_idx'].nunique()
    num_items = df_grouped['item_idx'].nunique()

    # Create implicit labels
    df_grouped['interaction'] = 1
    interactions = df_grouped[['customer_idx', 'item_idx', 'interaction']]

    # Train NeuMF
    class NeuMF(nn.Module):
        def __init__(self, num_users, num_items, emb_size=32):
            super().__init__()
            self.user_emb = nn.Embedding(num_users, emb_size)
            self.item_emb = nn.Embedding(num_items, emb_size)
            self.fc1 = nn.Linear(emb_size * 2, 64)
            self.fc2 = nn.Linear(64, 32)
            self.out = nn.Linear(32, 1)

        def forward(self, x):
            u = self.user_emb(x[:, 0])
            i = self.item_emb(x[:, 1])
            x = torch.cat([u, i], dim=1)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return torch.sigmoid(self.out(x)).squeeze()

    model = NeuMF(num_users, num_items)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()

    X = torch.LongTensor(interactions[['customer_idx', 'item_idx']].values)
    y = torch.FloatTensor(interactions['interaction'].values)

    with st.spinner("Training NeuMF..."):
        for epoch in range(5):
            model.train()
            optimizer.zero_grad()
            preds = model(X)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()

    st.success("✅ NeuMF trained!")

    # Customer selection
    selected_id = st.selectbox("Choose a CustomerID to recommend for", df['CustomerID'].unique())
    encoded_id = customer_encoder.transform([str(selected_id)])[0]
    item_indices = torch.arange(num_items)
    user_item_pairs = torch.column_stack((torch.full_like(item_indices, encoded_id), item_indices))

    model.eval()
    with torch.no_grad():
        scores = model(user_item_pairs).numpy()

    top_k = 10
    top_indices = scores.argsort()[-top_k:][::-1]
    recommended_codes = item_encoder.inverse_transform(top_indices)

    result_df = df_product[df_product['StockCode'].isin(recommended_codes)][
        ['StockCode', 'Product Name', 'Category', 'Brand', 'Unit Price']
    ].drop_duplicates().reset_index(drop=True)

    st.subheader(f"🧠 Top {top_k} Recommendations for Customer {selected_id} (NeuMF)")
    st.dataframe(result_df)

    st.markdown("🚀 *GCNN implementation can be added as advanced step using PyTorch Geometric.*")
