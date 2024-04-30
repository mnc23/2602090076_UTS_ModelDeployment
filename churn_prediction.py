import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

class ModelHandler:
    def __init__(self):
        self.model = None
        self.encoder = None
        self.scaler = None

    def load_model(self, model_path='XG_churn.pkl', encoder_path='one_hot_encode.pkl', scaler_path='scaling.pkl'):
        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)
        with open(encoder_path, 'rb') as file:
            self.encoder = pickle.load(file)
        with open(scaler_path, 'rb') as file:
            self.scaler = pickle.load(file)

    def preprocess_data(self, input_data):
        num_data = input_data.select_dtypes(['float64', 'int64']).columns
        obj_data = input_data.drop(num_data, axis=1).columns

        for col in obj_data:
            enc_data = self.encoder.transform(input_data[[col]])
            input_data = pd.concat([input_data.drop(columns=[col]), pd.DataFrame(enc_data.toarray(), columns=self.encoder.get_feature_names_out([col]))], axis=1)

        scaled_col = ['CustomerId', 'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
        input_data[scaled_cols] = self.scaler.transform(input_data[scaled_col])

        return input_data

    def predict_churn(self, input_data):
        preprocessed_data = self.preprocess_data(input_data)
        pred = self.model.predict(preprocessed_data)
        return pred

def main():
    st.title("Churn Prediction App")
    st.write("Churn Predict Detection.")

    # User input section
    credit_score = st.number_input("Credit Score:")
    geography = st.selectbox("Geography:", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender:", ["Male", "Female"])
    age = st.number_input("Age:")
    tenure = st.number_input("Tenure (years with company):")
    balance = st.number_input("Balance (account balance):")
    num_of_products = st.number_input("Number of Products:")
    has_credit_card = st.selectbox("Has Credit Card? (Yes/No)", ["Yes", "No"])
    is_active_member = st.selectbox("Is Active Member? (Yes/No)", ["Yes", "No"])
    estimated_salary = st.number_input("Estimated Salary:")

    input_data = pd.DataFrame({
        "CustomerId": [0],  # Placeholder for one-hot encoding
        "CreditScore": [credit_score],
        "Geography": [geography],
        "Gender": [gender],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [1 if has_credit_card == "Yes" else 0],
        "IsActiveMember": [1 if is_active_member == "Yes" else 0],
        "EstimatedSalary": [estimated_salary]
    })

    if st.button("Predict Churn Risk"):
        model_handler = ModelHandler()
        model_handler.load_model()
        prediction = model_handler.predict_churn(input_data)
        if prediction[0] == 1:
            st.write("There will be churn!")
        else:
            st.write("There will not be churn!")

if __name__ == "__main__":
    main()
