
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

# Load the regression model and preprocessing objects
model = tf.keras.models.load_model('ann_regression_salary_model.keras')

with open('regression_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('regression_label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('regression_onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

st.title("Customer Salary Prediction")

st.write("Enter customer details to predict Estimated Salary.")

credit_score = st.number_input("Credit Score", 300, 900, 650)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Female", "Male"])
age = st.number_input("Age", 18, 100, 35)
tenure = st.number_input("Tenure", 0, 10, 5)
balance = st.number_input("Balance", min_value=0.0, value=50000.0)
num_products = st.number_input("Number of Products", 1, 4, 2)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

if st.button("Predict Salary"):
    gender_encoded = label_encoder_gender.transform([gender])[0]

    geo_encoded = onehot_encoder_geo.transform(
        pd.DataFrame({"Geography": [geography]})
    )

    input_data = pd.DataFrame([[
        credit_score,
        gender_encoded,
        age,
        tenure,
        balance,
        num_products,
        has_cr_card,
        is_active_member,
        *geo_encoded[0]
    ]], columns=[
        "CreditScore", "Gender", "Age", "Tenure", "Balance",
        "NumOfProducts", "HasCrCard", "IsActiveMember",
        *onehot_encoder_geo.get_feature_names_out(["Geography"])
    ])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled, verbose=0)[0][0]

    st.success(f"Predicted Estimated Salary: {prediction:,.2f}")
