import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

class ModelHandler:
    def __init__(self):
        self.model = None
        self.scaler = None

    def load_model(self, model_path='XG_churn.pkl', scaler_path='scaling.pkl'):
        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)
        with open(scaler_path, 'rb') as file:
            self.scaler = pickle.load(file)

    def preprocess_data(self, input_data):
        scaled_col = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
        input_data[scaled_col] = self.scaler.transform(input_data[scaled_col])

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
    
    # Age input with validation for integers and maximum length of 3 digits
    age = st.number_input("Age:", step=1, format="%d", max_value=100)

    # Tenure slider with range 0-64
    tenure = st.slider("Tenure (years with company):", 0, 64)

    balance = st.number_input("Balance (account balance):")
    
    # NumOfProducts slider with range 1-4
    num_of_products = st.slider("Number of Products:", 1, 4)

    has_credit_card = st.selectbox("Has Credit Card? (Yes/No)", ["Yes", "No"])
    is_active_member = st.selectbox("Is Active Member? (Yes/No)", ["Yes", "No"])
    estimated_salary = st.number_input("Estimated Salary:")

    input_data = pd.DataFrame({
        "CreditScore": [credit_score],
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
