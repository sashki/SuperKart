import streamlit as st
import requests
import json

# Page configuration
st.set_page_config(page_title="SuperKart Sales Predictor", layout="centered")

st.title("🛒 SuperKart Sales Predictor")
st.markdown("Enter product and store details below to predict sales revenue.")

# Input fields
col1, col2 = st.columns(2)

with col1:
    product_weight = st.number_input("Product Weight", min_value=1.0, max_value=30.0, value=12.66, step=0.01)
    sugar_content = st.selectbox("Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
    allocated_area = st.number_input("Allocated Area", min_value=0.001, max_value=1.0, value=0.027, step=0.001, format="%.3f")
    product_mrp = st.number_input("Product MRP", min_value=10.0, max_value=500.0, value=117.08, step=0.01)

with col2:
    store_size = st.selectbox("Store Size", ["High", "Medium", "Small"])
    city_type = st.selectbox("City Type", ["Tier 1", "Tier 2", "Tier 3"])
    store_type = st.selectbox("Store Type", ["Departmental Store", "Supermarket Type1", "Supermarket Type2", "Food Mart"])
    store_age = st.number_input("Store Age (Years)", min_value=1, max_value=50, value=16, step=1)

product_type = st.selectbox("Product Category", ["Perishables", "Non Perishables"])

# API URL (update if running in Docker or Codespaces)
api_url = st.text_input("API URL", value="http://127.0.0.1:5000/predict")

# Predict button
if st.button("Predict Sales"):
    # Prepare payload
    payload = {
        "Product_Weight": product_weight,
        "Product_Sugar_Content": sugar_content,
        "Product_Allocated_Area": allocated_area,
        "Product_MRP": product_mrp,
        "Store_Size": store_size,
        "Store_Location_City_Type": city_type,
        "Store_Type": store_type,
        "Product_Id_char": "FD",  # Placeholder for Product_Id_char (not used in model)
        "Store_Age_Years": store_age,
        "Product_Type_Category": product_type
    }

    try:
        response = requests.post(api_url, json=payload)
        if response.status_code == 200:
            prediction = response.json().get("predicted_sales")
            st.success(f"✅ Predicted Sales: **${prediction:,.2f}**")
        else:
            st.error(f"❌ Error: {response.status_code} - {response.text}")
    except requests.exceptions.ConnectionError:
        st.error("❌ Could not connect to API. Make sure the Flask server is running at the specified URL.")