import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

# ==========================================
# 1. LOAD TRAINED MODEL
# ==========================================

model = tf.keras.models.load_model('ann_classification_model.h5')


# ==========================================
# 2. LOAD ENCODERS AND SCALER
# ==========================================

# Gender encoder
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

# Geography one-hot encoder
with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

# Feature scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


# ==========================================
# 3. STREAMLIT APP
# ==========================================

st.title('Customer Churn Prediction')


# ==========================================
# 4. USER INPUT
# ==========================================

geography = st.selectbox(
    'Geography',
    onehot_encoder_geo.categories_[0]
)

gender = st.selectbox(
    'Gender',
    label_encoder_gender.classes_
)

age = st.slider(
    'Age',
    18,
    92
)

balance = st.number_input(
    'Balance'
)

credit_score = st.number_input(
    'Credit Score'
)

estimated_salary = st.number_input(
    'Estimated Salary'
)

tenure = st.slider(
    'Tenure',
    0,
    10
)

num_of_products = st.slider(
    'Number of Products',
    1,
    4
)

has_cr_card = st.selectbox(
    'Has Credit Card',
    [0, 1]
)

is_active_member = st.selectbox(
    'Is Active Member',
    [0, 1]
)


# ==========================================
# 5. PREPARE INPUT DATA
# ==========================================

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})


# ==========================================
# 6. ONE-HOT ENCODE GEOGRAPHY
# ==========================================

geo_encoded = onehot_encoder_geo.transform(
    [[geography]]
).toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)


# ==========================================
# 7. COMBINE INPUT DATA
# ==========================================

input_data = pd.concat(
    [
        input_data.reset_index(drop=True),
        geo_encoded_df
    ],
    axis=1
)


# ==========================================
# 8. SCALE THE INPUT
# ==========================================

input_data_scaled = scaler.transform(input_data)


# ==========================================
# 9. MAKE PREDICTION
# ==========================================

prediction = model.predict(input_data_scaled)

prediction_proba = prediction[0][0]


# ==========================================
# 10. DISPLAY RESULT
# ==========================================

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')


# Display probability
st.write(
    f'Churn Probability: {prediction_proba:.2f}'
)

