# Feature preprocessing and data splitting
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler

# Machine Learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np

from datetime import date
from tqdm import tqdm
import os 


# Combine two datasets vertically (along columns) using pandas concat function.
def combine_two_datasets(dataset1,dataset2):
    '''
    Returns a  vertically concatenated dataset.
    Attributes:
    dataset1 - Dataset 1 to be combined 
    dataset2 - Dataset 2 to be combined
    '''
    
    data = pd.concat([dataset1,dataset2], axis=1)
    data = data.loc[:, ~data.columns.duplicated()]
    # filling nan values with the median of values if the column values are numeric
    numeric_columns = data.select_dtypes(include=['number']).columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())
    return data

def split_data(X, y, test_size=0.3, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def scale_data_return_dataframe(dataframe_train, dataframe_test, columns_to_remove = []): # columns_to_remove is a list of strings
    scaling_columns = dataframe_train.columns.drop(columns_to_remove)
    
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(dataframe_train[scaling_columns].values)
    
    scaled_array_Y = scaler.transform(dataframe_test[scaling_columns].values)
    
    df_scaled_features = pd.DataFrame(
    scaled_array,
    columns=scaling_columns,
    index=dataframe_train.index # Manter o mesmo índice para fácil concatenação
    )
    
    df_scaled_features_Y = pd.DataFrame(
    scaled_array_Y,
    columns=scaling_columns,
    index=dataframe_test.index # Manter o mesmo índice para fácil concatenação
    )
    
    df_excluded_columns = dataframe_train[columns_to_remove]
    
    df_excluded_columns_Y = dataframe_test[columns_to_remove]
    
    full_dataset_scaled_train = pd.concat(
    [df_excluded_columns, df_scaled_features], 
    axis=1
    )

    full_dataset_scaled_test = pd.concat(
    [df_excluded_columns_Y, df_scaled_features_Y],
    axis=1
    )
    
    return full_dataset_scaled_train, full_dataset_scaled_test, scaler

def apply_scale_data_return_dataframe(dataframe, already_defined_scaler): # columns_to_remove is a list of strings
    
    scaled_array = already_defined_scaler.transform(dataframe)
    
    df_scaled_features = pd.DataFrame(
    scaled_array,
    columns=dataframe.columns,
    index=dataframe.index # Manter o mesmo índice para fácil concatenação
    )
    
    return df_scaled_features

def evaluate_model(Y_predicted, y_true, dataset_name="Test"):    
    r2 = r2_score(y_true, Y_predicted)
    rmse = np.sqrt(mean_squared_error(y_true, Y_predicted))
    print(f"\n{dataset_name} Evaluation:")
    print(f"R²: {r2:.3f}")
    print(f"RMSE: {rmse:.3f}")
    return Y_predicted, r2, rmse


# esse aqui é só um modelo, n precisa ser assim nao.
def run_pipeline(X, y, param_name="Parameter"):
    print(f"\n{'='*60}")
    print(f"Training Model for {param_name}")
    print(f"{'='*60}")
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Scale
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
    
    # Train
    model = train_model(X_train_scaled, y_train)
    
    # Evaluate (in-sample)
    y_train_pred, r2_train, rmse_train = evaluate_model(model, X_train_scaled, y_train, "Train")
    
    # Evaluate (out-sample)
    y_test_pred, r2_test, rmse_test = evaluate_model(model, X_test_scaled, y_test, "Test")
    
    # Return summary
    results = {
        "Parameter": param_name,
        "R2_Train": r2_train,
        "RMSE_Train": rmse_train,
        "R2_Test": r2_test,
        "RMSE_Test": rmse_test
    }
    return model, scaler, pd.DataFrame([results])

class DataOrganizer:
    def __init__(self, target_columns: list):
        self.training_loaded = False
        self.full_training_dataset = None
        self.full_submission_dataset = None
        self.feature_columns = None
        self.target_columns = target_columns
        self.feature_training_dataset = None
        self.target_training_datasets = {}
        self.feature_submission_dataset = None
        self.scaler = None
        
    def load_training_data(self, csv_files: list, drop_from_feature_columns: list, scale: bool = False):
        csv_data = []
        for file in csv_files:
            data = pd.read_csv(file)
            csv_data.append(data)
        
        self.full_training_dataset = None
        for data in csv_data:
            self.full_training_dataset = combine_two_datasets(self.full_training_dataset, data)
        
        # 1. Adicionar as features cíclicas
        self.full_training_dataset = self.add_cyclical_features(self.full_training_dataset, date_column='Sample Date')
        
        # 2. DEFINIR A "LISTA NEGRA" DE COLUNAS (O que NUNCA deve ser feature)
        # Forçamos a remoção de IDs e dados temporais brutos
        forbidden_cols = ["Latitude", "Longitude", "Sample Date", "Year", "MonthOfYear"]
        
        # 3. Criar a lista de colunas que serão usadas no treino
        # Elas não podem estar nos targets, nem na lista de drop do usuário, nem na lista negra
        self.feature_columns = [
            col for col in self.full_training_dataset.columns 
            if col not in self.target_columns and 
               col not in drop_from_feature_columns and 
               col not in forbidden_cols
        ]
        
        # 4. Extrair o dataset final de treino
        self.feature_training_dataset = self.full_training_dataset[self.feature_columns]
        
        # Escalonamento (se ativado)
        if scale:
            self.scaler = StandardScaler()
            scaled_values = self.scaler.fit_transform(self.feature_training_dataset.values)
            self.feature_training_dataset = pd.DataFrame(
                scaled_values, columns=self.feature_columns, index=self.feature_training_dataset.index
            )
        
        self.target_training_datasets = {target: self.full_training_dataset[target] for target in self.target_columns}
        self.training_loaded = True

    def load_submission_data(self, csv_files: list):
        if not self.training_loaded:
            raise Exception("Training data must be loaded before loading submission data.")
        
        csv_data = []
        for file in csv_files:
            data = pd.read_csv(file)
            csv_data.append(data)
        
        self.full_submission_dataset = None
        for data in csv_data:
            self.full_submission_dataset = combine_two_datasets(self.full_submission_dataset, data)
        
        self.full_submission_dataset = self.add_cyclical_features(self.full_submission_dataset, date_column='Sample Date')
        
        # 5. GARANTIR que as colunas de submissão sejam EXATAMENTE as mesmas do treino
        # e na mesma ordem!
        self.feature_submission_dataset = self.full_submission_dataset[self.feature_columns]
        
        if self.scaler is not None:
            scaled_values = self.scaler.transform(self.feature_submission_dataset.values)
            self.feature_submission_dataset = pd.DataFrame(
                scaled_values, columns=self.feature_columns, index=self.feature_submission_dataset.index
            )
    def get_training_dataset(self):
        return self.feature_training_dataset, self.target_training_datasets
    
    def get_submission_dataset(self):
        return self.feature_submission_dataset
    
    def get_feature_columns(self):
        return self.feature_columns
    
    def get_full_training_dataset(self):
        return self.full_training_dataset
    
    def build_get_submission_dataset(self, predicted_values_dict: dict):
        if not self.training_loaded:
            raise Exception("Training data must be loaded before building submission dataset.")
        
        loc_and_time_data = pd.DataFrame({
            'Latitude': self.full_submission_dataset['Latitude'],
            'Longitude': self.full_submission_dataset['Longitude'],
            'Sample Date': self.full_submission_dataset['Sample Date'].dt.strftime("%d-%m-%Y"),
        })
        
        predicted_values_pd = pd.DataFrame(predicted_values_dict)
        
        return_pd = pd.concat([loc_and_time_data, predicted_values_pd], axis=1)
        
        return return_pd
    
    def add_cyclical_features(self, data_variable, date_column='Sample Date'):
        # Converte para datetime de forma robusta
        data_variable[date_column] = pd.to_datetime(data_variable[date_column], dayfirst=True)
        
        # Cria apenas variáveis numéricas derivadas da data
        data_variable['MonthOfYear'] = data_variable[date_column].dt.month
        data_variable['month_sin'] = np.sin(2 * np.pi * data_variable['MonthOfYear'] / 12)
        data_variable['Year'] = data_variable[date_column].dt.year
        
        return data_variable