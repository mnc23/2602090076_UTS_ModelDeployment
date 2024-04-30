import pandas as pd
import numpy as np
import seaborn as sns
import pickle as pkl
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle as pkl

import warnings
warnings.filterwarnings('ignore')

class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.input_df = None
        self.output_df = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)

    def fill_missing_values(self):
        filled_data = self.data.copy()
        for column in filled_data.columns:
            if filled_data[column].dtype == 'object':
                # Jika tipe data string, isi dengan modus
                mode_val = filled_data[column].mode()[0]
                filled_data[column].fillna(mode_val, inplace=True)
            else:
                # Jika tipe data numerik, isi dengan rata-rata
                mean_val = filled_data[column].mean()
                filled_data[column].fillna(mean_val, inplace=True)
        return filled_data

    def split_input_output(self, target_column):
        self.output_df = self.data[target_column]
        self.input_df = self.data.drop([target_column], axis=1)

  
class ModelHandler:
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.x_train, self.x_test, self.y_train, self.y_test, self.y_predict = [None] * 5
    
    def drop_column(self, colums):
        self.input_data = self.input_data.drop(colums, axis=1)

    def split_data(self, test_size=0.2, random_state=42):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.input_data, self.output_data, test_size=test_size, random_state=random_state)
        
    def one_hot_encode(self):
        num_data = self.input_data.select_dtypes(['float64', 'int64']).columns
        obj_data = self.input_data.drop(num_data, axis=1).columns

        for col in obj_data:
            encoder = OneHotEncoder(handle_unknown="ignore")
            enc_train = encoder.fit_transform(self.x_train[[col]])
            enc_test = encoder.transform(self.x_test[[col]])

            col_to_drop = [col]

            # Reset indices before concatenation
            self.x_train = self.x_train.reset_index(drop=True)
            self.x_test = self.x_test.reset_index(drop=True)
            self.x_train = pd.concat([self.x_train.drop(columns=col_to_drop), pd.DataFrame(enc_train.toarray(), columns=encoder.get_feature_names_out(col_to_drop))], axis=1)
            self.x_test = pd.concat([self.x_test.drop(columns=col_to_drop), pd.DataFrame(enc_test.toarray(), columns=encoder.get_feature_names_out(col_to_drop))], axis=1)

        filename = 'one_hot_encode.pkl'
        pkl.dump(encoder, open(filename, 'wb'))

      
    def scale_data(self, scaled_cols):
        std_scaler = StandardScaler()

        # Lakukan standarisasi pada data latih
        x_train_scaled_temp = std_scaler.fit_transform(self.x_train[scaled_cols])
        self.x_train[scaled_cols] = pd.DataFrame(x_train_scaled_temp, columns=scaled_cols)

        # Lakukan standarisasi pada data uji
        x_test_scaled_temp = std_scaler.transform(self.x_test[scaled_cols])
        self.x_test[scaled_cols] = pd.DataFrame(x_test_scaled_temp, columns=scaled_cols)

        filename = 'scaling.pkl'
        pkl.dump(std_scaler, open(filename, 'wb'))

    def createModel(self):
          self.model = XGBClassifier(
            n_estimators = 100,
            max_depth = 3,
            learning_rate = 0.1
          )
      
    def train(self):
        self.model.fit(self.x_train, self.y_train)
              
    def predict(self):
          self.y_pred =  self.model.predict(self.x_test)
          
    def createReport(self):
          print('\nClassification Report\n')
          print(classification_report(self.y_test, self.y_pred))

    def evaluate_model(self):
          predict = self.model.predict(self.x_test)
          return accuracy_score(self.y_test, predict)
      
    def pickle_dump(self, filename='XG_churn.pkl'):
          with open(filename, 'wb') as file:
              pkl.dump(self.model, file)


file_path = 'data_D.csv'
data_handler = DataHandler(file_path)
data_handler.load_data()
data_handler.split_input_output('churn')
data_handler.fill_missing_values()
    
input_df = data_handler.input_df
output_df = data_handler.output_df

model_handler = ModelHandler(input_df, output_df)
model_handler.drop_column(['Unnamed: 0', 'id', 'Surname'])
    
model_handler.split_data()

model_handler.one_hot_encode()

model_handler.scale_data(['CustomerId', 'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary'])

model_handler.createModel()
model_handler.train()
model_handler.predict()
print("Accuracy: ", model_handler.evaluate_model())
model_handler.createReport()

model_handler.pickle_dump()

