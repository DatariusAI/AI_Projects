# app.py

# Imports
import streamlit as st
import pandas as pd
import numpy as np

# Attempt to import matplotlib
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    is_matplotlib_available = True
except ImportError as e:
    st.warning(f"Matplotlib is not installed. Some features might not work. Error: {e}")
    is_matplotlib_available = False

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
import os
import qrcode
from PIL import Image

# Additional imports and setup
is_torch_available = False
is_tf_available = False
is_sentiment_available = False

# Check for TensorFlow and PyTorch
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
os.environ["OPENAI_API_KEY"] = "API Key"

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
st.title("Ambiscions - Case Study")

# QR Codes
st.header("Access the Streamlit App")
col1, col2 = st.columns(2)

# Generate QR Code for localhost
local_url = "http://localhost:8501"
local_qr = qrcode.make(local_url)
local_qr_image = local_qr.resize((150, 150))

# Generate QR Code for deployed URL
deployed_url = "https://your-deployed-url.com"  # Replace with your actual deployed URL
deployed_qr = qrcode.make(deployed_url)
deployed_qr_image = deployed_qr.resize((150, 150))

with col1:
    st.image(local_qr_image, caption="Localhost URL QR Code")

with col2:
    st.image(deployed_qr_image, caption="Deployed URL QR Code")

# Check if Matplotlib is available
if is_matplotlib_available:
    # Descriptive Analysis and plotting code
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

# Continue with other sections of your Streamlit app...
