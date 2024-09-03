# app.py

# Imports
import streamlit as st
import pandas as pd
import numpy as np
import os

# Attempt to import matplotlib and seaborn for plotting
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    is_matplotlib_available = True
except ImportError as e:
    st.warning(f"Matplotlib or Seaborn is not installed. Some features might not work. Error: {e}")
    is_matplotlib_available = False

# Other necessary imports
try:
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.cluster import KMeans
    from sklearn.metrics import mean_squared_error
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    is_sklearn_xgb_available = True
except ImportError as e:
    st.warning(f"Scikit-learn or XGBoost is not installed. Some features might not work. Error: {e}")
    is_sklearn_xgb_available = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    is_plotly_available = True
except ImportError as e:
    st.warning(f"Plotly is not installed. Some features might not work. Error: {e}")
    is_plotly_available = False

# Additional imports for QR code generation
try:
    import qrcode
    from PIL import Image
    is_qrcode_available = True
except ImportError as e:
    st.warning(f"qrcode or PIL is not installed. QR code features will not work. Error: {e}")
    is_qrcode_available = False

# Import TensorFlow and PyTorch conditionally for sentiment analysis
is_torch_available = False
is_tf_available = False
is_sentiment_available = False

try:
    import tensorflow as tf
    is_tf_available = True
except ImportError:
    pass

try:
    import torch
    is_torch_available = True
except (OSError, ImportError):
    is_torch_available = False

if is_torch_available or is_tf_available:
    try:
        from transformers import pipeline
        classifier = pipeline('sentiment-analysis')
        is_sentiment_available = True
    except Exception:
        is_sentiment_available = False

# Set API key for OpenAI (replace 'your-api-key' with your actual API key)
os.environ["OPENAI_API_KEY"] = "your-api-key"

# Load datasets
suffix = "https://drive.google.com/uc?id="
sales_data = pd.read_csv(suffix + "1Lj7Zke3LHCOAqPIRwFJOrZeUbkMhyEfB", encoding='utf8')
promotion_data = pd.read_csv(suffix + "1idK_ctZD72TDWXy10qymhQ308qniZCvH", encoding='utf8')
customer_data = pd.read_csv(suffix + "18n8qug_i4OvRzFo1E0-pPBU6L038PH6L", encoding='utf8')
product_data = pd.read_csv(suffix + "1NYaVT8pnypvqGRGweiwwP7TrqR4cMPNn", encoding='utf8')
store_data = pd.read_csv(suffix + "1LIuZxAsBiEgNT0XfWkhM_YkPohEO_8uy", encoding='utf8')

# Data processing
sales_data['Transaction_Date'] = pd.to_datetime(sales_data['Transaction_Date'])
customer_data['First_Purchase_Date'] = pd.to_datetime(customer_data['First_Purchase_Date'])
sales_data['Transaction_Date_Copy'] = sales_data['Transaction_Date']
sales_data.set_index('Transaction_Date', inplace=True)
sales_data.sort_index(inplace=True)
sales_data['Transaction_Date'] = sales_data['Transaction_Date_Copy']
sales_data['Quantity_Sold'] = sales_data['Quantity_Sold'].clip(upper=500)
sales_data['Total_Amount'] = sales_data['Quantity_Sold'] * sales_data['Price_Per_Unit']

# Streamlit title
st.title("Ambiscions - Case Study: Test Pilot")

# QR Code section
if is_qrcode_available:
    st.header("Test Pilot App")
    col1, col2 = st.columns(2)

    # Generate QR Code for localhost
    local_url = "http://172.20.25.97:8501"
    local_qr = qrcode.make(local_url)
    local_qr_image = local_qr.resize((150, 150))

    # Generate QR Code for deployed URL
    deployed_url = "http://172.20.25.97:8501"  # Replace with your actual deployed URL
    deployed_qr = qrcode.make(deployed_url)
    deployed_qr_image = deployed_qr.resize((150, 150))

    with col1:
        st.image(local_qr_image, caption="Localhost URL QR Code")

    with col2:
        st.image(deployed_qr_image, caption="Deployed URL QR Code")

# Descriptive Analysis if Matplotlib is available
if is_matplotlib_available:
    st.header("1. Descriptive Analysis")
    merged_data = pd.merge(sales_data, store_data, on='Store_ID', how='left')
    avg_sales_per_location = merged_data.groupby('Location')['Total_Amount'].mean().reset_index()
    sales_data['Transaction_Month'] = sales_data['Transaction_Date'].dt.to_period('M')
    avg_sales_over_time = sales_data.groupby('Transaction_Month')['Quantity_Sold'].mean().rolling(2).mean().reset_index()
    avg_sales_over_time['Transaction_Month'] = avg_sales_over_time['Transaction_Month'].dt.to_timestamp()
    merged_data = pd.merge(sales_data, customer_data, on='Customer_ID', how='left')
    avg_transaction_value_by_age = merged_data.groupby('Age')['Quantity_Sold'].mean().reset_index()
    num_young_customers_over_time = customer_data[customer_data['Age'] < 35].groupby(customer_data['First_Purchase_Date'].dt.to_period('M')).size().reset_index(name='count')
    num_young_customers_over_time['First_Purchase_Date'] = num_young_customers_over_time['First_Purchase_Date'].dt.to_timestamp()

    fig, axs = plt.subplots(2, 2, figsize=(20, 10))
    sns.barplot(x='Location', y='Total_Amount', data=avg_sales_per_location, ax=axs[0, 0])
    axs[0, 0].set_title('Average Sales per Store')
    sns.lineplot(x='Transaction_Month', y='Quantity_Sold', data=avg_sales_over_time, ax=axs[0, 1])
    axs[0, 1].set_title('Average Sales over Time (moving average)')
    sns.lineplot(x='Age', y='Quantity_Sold', data=avg_transaction_value_by_age, ax=axs[1, 0])
    axs[1, 0].set_title('Average Basket by Age')
    sns.lineplot(x='First_Purchase_Date', y='count', data=num_young_customers_over_time, ax=axs[1, 1])
    axs[1, 1].set_title('Number of Young Customers over Time')
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.warning("Matplotlib is not available. Descriptive Analysis plots will not be displayed.")

# Customer Segmentation if sklearn and XGBoost are available
if is_sklearn_xgb_available:
    st.subheader("Machine Learning - Customer Segmentation")

    customer_sales = sales_data.groupby('Customer_ID').agg({'Quantity_Sold':'sum'}).reset_index()
    customer_data_copy = pd.merge(customer_data.copy(), customer_sales, on='Customer_ID')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['Age', 'Quantity_Sold']),
            ('cat', OneHotEncoder(), ['Gender', 'Income_Level'])])

    customer_data_preprocessed = preprocessor.fit_transform(customer_data_copy)

    num_clusters = st.slider("Select number of clusters", 1, 10, 1)
    kmeans = KMeans(n_clusters=num_clusters)
    customer_data_copy['Cluster'] = kmeans.fit_predict(customer_data_preprocessed)

    x_var = st.selectbox('Select X Variable', ['Age', 'Gender', 'Income_Level', 'First_Purchase_Date', 'Quantity_Sold'])
    y_var = st.selectbox('Select Y Variable', ['Age', 'Gender', 'Income_Level', 'First_Purchase_Date', 'Quantity_Sold'])

    if is_plotly_available:
        fig = px.scatter(customer_data_copy, x=x_var, y=y_var, color='Cluster', category_orders={'Cluster': list(range(num_clusters))})
        st.plotly_chart(fig)
    else:
        st.warning("Plotly is not available. Segmentation plots will not be displayed.")
else:
    st.warning("Scikit-Learn or XGBoost is not available. Customer Segmentation will not be displayed.")

# Sentiment Analysis if sentiment model is available
st.subheader("Machine Learning - Sentiment Analysis")

if is_sentiment_available:
    user_input = st.text_input("Enter a sentence to analyze sentiment:")
    if user_input:
        def get_sentiment(text):
            result = classifier(text)[0]
            if result['score'] < 0.85:
                return 0
            elif result['label'] == 'POSITIVE':
                return 1
            else:
                return -1
        sentiment = get_sentiment(user_input)
        st.write(f"Predicted sentiment score: {sentiment}")
else:
    st.info("Sentiment analysis is currently unavailable. Please ensure that the required dependencies are installed.")

# Visualization - Items Sold Per Sentiment
if is_matplotlib_available:
    st.subheader("Items Sold Per Sentiment")
    plt.figure(figsize=(10, 6))
    merged_data = pd.merge(sales_data, product_data, on='Product_ID', how='left')
    sns.violinplot(x='Sentiment', y='Quantity_Sold', data=merged_data)
    plt.title('Items Sold Per Sentiment')
    st.pyplot(plt)
else:
    st.warning("Matplotlib is not available. Items Sold Per Sentiment plot will not be displayed.")

# Diagnostic Analysis - Discounts and Sales
st.header("2. Diagnostic Analysis")
st.subheader("Discounts and Sales")

if is_matplotlib_available:
    sales_data['Transaction_Month'] = sales_data['Transaction_Date'].dt.to_period('M')
    monthly_sales = sales_data.groupby('Transaction_Month').agg({'Quantity_Sold': 'sum', 'Total_Amount': 'sum', 'Transaction_Date': 'min'}).reset_index()
    monthly_sales['Transaction_Month'] = monthly_sales['Transaction_Month'].dt.to_timestamp()

    promotion_data['Start_Date'] = pd.to_datetime(promotion_data['Start_Date'])
    promotion_data['End_Date'] = pd.to_datetime(promotion_data['End_Date'])

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(monthly_sales['Transaction_Date'], monthly_sales['Quantity_Sold'], label='Sales')

    for i in range(len(promotion_data)):
        ax.axvspan(promotion_data.loc[i, 'Start_Date'], promotion_data.loc[i, 'End_Date'], color='red', alpha=0.3)

    ax.xaxis.set_major_locator(plt.MaxNLocator())
    plt.gcf().autofmt_xdate()
    plt.ylabel('Items Sold')
    plt.title('Items Sold, per Month, Indicating Discounts')
    plt.legend()
    plt.grid()
    st.pyplot(fig)
else:
    st.warning("Matplotlib is not available. Discounts and Sales plot will not be displayed.")

# Correlation Between Variables if Plotly is available
st.subheader("Correlation Between Variables")
if is_plotly_available:
    df = sales_data.merge(product_data, on='Product_ID', how='left')
    df = df.merge(customer_data, on='Customer_ID', how='left')
    df = df.merge(store_data, on='Store_ID', how='left')
    df = df.merge(promotion_data, on='Promotion_ID', how='left')
    df["Discount"] = df.Discount.fillna(0)

    selected_variables = st.multiselect('Select variables for correlation matrix', df.columns.tolist(), default=df.columns[:5].tolist())
    if selected_variables:
        corr_matrix = df[selected_variables].corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto")
        st.plotly_chart(fig)
else:
    st.warning("Plotly is not available. Correlation plots will not be displayed.")

# Contribution of Artificial Intelligence - Auto-GPT
st.subheader("Contribution of Artificial Intelligence - Auto-GPT")

# Merging dataframes for analysis
ddf = sales_data.merge(product_data, on='Product_ID', how='left')
ddf = ddf.merge(customer_data, on='Customer_ID', how='left')

query_input = st.text_area("Enter your query for the dataset", "What factors are contributing to sales decline?")
if st.button("Analyze"):
    st.write("Analyzing the dataset...")

    # Simulated LLM response
    llm_response = "The decline in sales seems to be primarily driven by a reduction in the number of new customers and a decrease in overall store traffic. Further analysis suggests that the recent marketing campaigns have not been effective in attracting the target demographic."
    
    # Display LLM result
    st.write("LLM Analysis Result:")
    st.write(llm_response)

# Predictive Analysis
st.header("3. Predictive Analysis")

if is_sklearn_xgb_available:
    data = sales_data.merge(product_data, on="Product_ID")\
                     .merge(customer_data, on="Customer_ID")\
                     .merge(store_data, on="Store_ID")\
                     .merge(promotion_data, on="Promotion_ID")

    merged_data = data.copy()
    merged_data["Discount"] = merged_data["Discount"].fillna(0)

    merged_data['Transaction_Year'] = merged_data['Transaction_Date'].dt.year
    merged_data['Transaction_Month'] = merged_data['Transaction_Date'].dt.month

    merged_data = pd.get_dummies(merged_data, columns=['Category', 'Gender', 'Income_Level', 'Location'], drop_first=True)
    merged_data = merged_data.select_dtypes(include=[np.number])

    target = merged_data['Quantity_Sold']
    features = merged_data.drop(columns=['Quantity_Sold'])

    X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(objective='reg:squarederror', learning_rate=0.1, max_depth=5, n_estimators=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    st.write("Root Mean Square Error (RMSE):", rmse)

    st.subheader("Feature Importance")
    fig, ax = plt.subplots()
    xgb.plot_importance(model, ax=ax)
    st.pyplot(fig)
else:
    st.warning("Scikit-Learn or XGBoost is not available. Predictive Analysis will not be displayed.")

# Model Creation
st.header("4. Prescriptive Analysis")
st.subheader("Model Creation")

if is_sklearn_xgb_available:
    data = sales_data.merge(product_data, on="Product_ID")\
                     .merge(customer_data, on="Customer_ID")\
                     .merge(store_data, on="Store_ID")\
                     .merge(promotion_data, on="Promotion_ID")

    merged_data = data.copy()
    merged_data["Discount"] = merged_data["Discount"].fillna(0)

    merged_data['Transaction_Year'] = merged_data['Transaction_Date'].dt.year
    merged_data['Transaction_Month'] = merged_data['Transaction_Date'].dt.month

    try:
        if 'Gender' in merged_data.columns and 'Income_Level' in merged_data.columns:
            merged_data = pd.get_dummies(merged_data, columns=['Gender', 'Income_Level'])
        else:
            st.warning("Columns 'Gender' and 'Income_Level' not found in the DataFrame for one-hot encoding.")
    except Exception as e:
        st.error(f"An error occurred during one-hot encoding: {e}")

    X = merged_data[['Age', 'Discount', 'Transaction_Year', 'Transaction_Month', 'Gender_Male', 'Gender_Female', 'Income_Level_Low', 'Income_Level_Medium', 'Income_Level_High']]
    y = merged_data['Quantity_Sold']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    model = xgb.XGBRegressor(objective ='reg:squarederror', learning_rate = 0.1, max_depth = 5, n_estimators = 100)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    st.subheader("Optimize Predictions")

    st.sidebar.header("Adjust Parameters for Prediction")
    age = st.sidebar.slider('Average Customer Age', int(merged_data['Age'].min()), int(merged_data['Age'].max()), int(merged_data['Age'].mean()))
    sentiment = st.sidebar.slider('Average Sentiment', -1.0, 1.0, 0.0, 0.1)
    discount = st.sidebar.slider('Average Discount', float(merged_data['Discount'].min()), float(merged_data['Discount'].max()), float(merged_data['Discount'].mean()), 0.01)
    gender = st.sidebar.selectbox('Customer Gender', ['Male', 'Female'])
    income = st.sidebar.selectbox('Customer Assessed Wealth Bracket', ['Low', 'Medium', 'High'])

    last_date = sales_data['Transaction_Date'].max()
    future_dates = pd.date_range(start=last_date, periods=12*30)
    future_data = pd.DataFrame(future_dates, columns=['Transaction_Date'])

    future_data['Sentiment'] = sentiment
    future_data['Age'] = age
    future_data['Discount'] = discount
    future_data['Transaction_Year'] = future_data['Transaction_Date'].dt.year
    future_data['Transaction_Month'] = future_data['Transaction_Date'].dt.month
    future_data['Gender_Male'] = 1 if gender == 'Male' else 0
    future_data['Gender_Female'] = 1 if gender == 'Female' else 0
    future_data['Income_Level_Low'] = 1 if income == 'Low' else 0
    future_data['Income_Level_Medium'] = 1 if income == 'Medium' else 0
    future_data['Income_Level_High'] = 1 if income == 'High' else 0

    future_data = future_data[X_train.columns]

    future_predictions = model.predict(future_data)
    future_data["Transaction_Date"] = future_dates
    future_data['Predicted_Sales'] = future_predictions

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=future_data['Transaction_Date'], y=future_data['Predicted_Sales'], mode='lines', name='Predicted Sales'))

    fig.update_layout(
        title='Predicted Sales Over the Next Six Months',
        xaxis_title='Date',
        yaxis_title='Predicted Sales'
    )

    st.plotly_chart(fig)
else:
    st.warning("Scikit-Learn or XGBoost is not available. Model creation and optimization will not be displayed.")
