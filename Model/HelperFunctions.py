# Feature preprocessing and data splitting
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler

# Machine Learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

from sklearn.base import clone
from tqdm.auto import tqdm

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

def evaluate_model(Y_predicted, y_true, dataset_name="Test", verbose = False):    
    r2 = r2_score(y_true, Y_predicted)
    rmse = np.sqrt(mean_squared_error(y_true, Y_predicted))
    if verbose:
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

def get_real_signals(model, X_val, y_val, target_name):
    # Calcula a importância por permutação
    result = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=42, n_jobs=-1)
    
    # Cria um DataFrame com os resultados
    perm_df = pd.DataFrame({
        'feature': X_val.columns,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    }).sort_values(by='importance_mean', ascending=False)
    
    print(f"--- Sinais Reais para {target_name} ---")
    print(perm_df.head(10))
    return perm_df


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


def get_location_train_test_split(dataHandler: DataOrganizer, test_size=0.2, random_state=42):
    """
    Realiza o split de treino e teste garantindo que locais (Lat/Lon) 
    inteiros fiquem apenas no treino ou apenas no teste.
    """
    # 1. Extrair os dados brutos do handler
    Feature_data, Target_data = dataHandler.get_training_dataset()
    full_training_data = dataHandler.get_full_training_dataset()
    
    # 2. Criar os grupos baseados na localização (Chave única para cada ponto no mapa)
    groups = full_training_data[["Latitude", "Longitude"]].astype(str).agg('|'.join, axis=1).values 
    
    # 3. Configurar o Splitter
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    
    # Como as coordenadas são as mesmas para todos os targets, 
    # podemos gerar os índices uma única vez para garantir consistência.
    # Usamos o primeiro target da lista como referência para o split.
    any_target = list(Target_data.values())[0]
    train_idx, test_idx = next(gss.split(Feature_data, any_target, groups=groups))
    
    # 4. Organizar os dados em um dicionário estruturado
    split_results = {}
    
    for target_name in Target_data.keys():
        split_results[target_name] = {
            'X_train': Feature_data.iloc[train_idx],
            'X_test':  Feature_data.iloc[test_idx],
            'Y_train': Target_data[target_name].iloc[train_idx],
            'Y_test':  Target_data[target_name].iloc[test_idx]
        }
    
    print(f"✅ Split concluído: {len(train_idx)} amostras para treino, {len(test_idx)} para teste.")
    return split_results

    
    
class FeatureClassifier:
    def __init__(self, csv_training_files: list):
        self.csv_training_files = csv_training_files
        self.target_columns = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
        
        print("📥 Carregando todos os dados para memória...")
        self.dataHandler = DataOrganizer(self.target_columns)
        self.dataHandler.load_training_data(self.csv_training_files, drop_from_feature_columns=[], scale=False)
        
        self.all_possible_features = self.dataHandler.get_feature_columns()
        self.model = None
        
        # Splits
        self.splits = get_location_train_test_split(self.dataHandler, test_size=0.2)
    
    def define_model(self, model):
        self.model = model
    
    def classify_features_per_target(self, feature_to_start='pet', penalty_weight=0.5):
        """
        penalty_weight: Quanto penalizar o overfitting.
        0.0 = Só importa o Teste (seu código antigo).
        0.5 = Balanceado (Recomendado).
        1.0 = Conservador (Prioriza modelos onde Treino e Teste são iguais).
        """
        
        if self.model is None:
            raise Exception("❌ Erro: Defina o modelo usando define_model() antes.")

        final_results = {target: [] for target in self.target_columns}
        
        for target in self.target_columns:
            print(f"\n🎯 Seleção para {target} (Penalidade: {penalty_weight})")
            
            X_train_full = self.splits[target]['X_train']
            X_test_full  = self.splits[target]['X_test']
            y_train      = self.splits[target]['Y_train']
            y_test       = self.splits[target]['Y_test']
            
            # --- CONFIGURAÇÃO INICIAL ---
            if feature_to_start in self.all_possible_features:
                features_to_maintain = [feature_to_start]
            else:
                features_to_maintain = []
            
            best_quality_score = -np.inf
            best_r2_test_ref = -np.inf
            
            # --- AVALIAR BASELINE (Feature Inicial) ---
            if features_to_maintain:
                model_clone = clone(self.model)
                model_clone.fit(X_train_full[features_to_maintain], y_train)
                
                # Predição Treino (Para calcular o GAP)
                y_pred_train = model_clone.predict(X_train_full[features_to_maintain])
                _, r2_train, _ = evaluate_model(y_pred_train, y_train, dataset_name="Train", verbose=False)
                
                # Predição Teste
                y_pred_test = model_clone.predict(X_test_full[features_to_maintain])
                _, r2_test, _ = evaluate_model(y_pred_test, y_test, dataset_name="Baseline", verbose=False)
                
                # CÁLCULO DA MÉTRICA COMPOSTA
                gap = abs(r2_train - r2_test)
                best_quality_score = r2_test - (penalty_weight * gap)
                best_r2_test_ref = r2_test # Apenas para referência visual
                
                print(f"   🔹 Base Score: {best_quality_score:.4f} (R2 Test: {r2_test:.3f} | Gap: {gap:.3f})")
            
            candidates = [f for f in self.all_possible_features if f not in features_to_maintain]
            
            # --- LOOP DE SELEÇÃO ---
            for feature in tqdm(candidates):
                current_features = features_to_maintain + [feature]
                
                X_train_subset = X_train_full[current_features]
                X_test_subset = X_test_full[current_features]

                model_clone = clone(self.model)
                model_clone.fit(X_train_subset, y_train)
                
                # 1. Avaliar Treino (Necessário para ver Overfitting)
                y_train_pred = model_clone.predict(X_train_subset)
                # (Supondo que evaluate_model retorna y_pred, r2, rmse)
                _, r2_train, _ = evaluate_model(y_train_pred, y_train, dataset_name="T", verbose=False)
                
                # 2. Avaliar Teste
                y_test_pred = model_clone.predict(X_test_subset)
                _, r2_test, _ = evaluate_model(y_test_pred, y_test, dataset_name="T", verbose=False)
                
                # 3. Calcular Score Penalizado
                # Se o treino for muito melhor que o teste, o gap cresce e o score cai
                gap = abs(r2_train - r2_test)
                current_quality_score = r2_test - (penalty_weight * gap)
                
                # 4. Decisão
                if current_quality_score > best_quality_score:
                    features_to_maintain.append(feature)
                    best_quality_score = current_quality_score
                    best_r2_test_ref = r2_test
                    # Opcional: Mostrar quando melhora
                    # print(f"   ✅ + {feature} (Score: {current_quality_score:.4f} | R2: {r2_test:.3f} | Gap: {gap:.3f})")
                
            final_results[target] = features_to_maintain
            print(f"🏁 Finalizado para {target}")
            print(f"   Features: {len(features_to_maintain)}")
            print(f"   Melhor Score Penalizado: {best_quality_score:.4f}")
            print(f"   R² Teste Final: {best_r2_test_ref:.4f}")
            print(f"   Lista: {features_to_maintain}")

        return final_results
    
### Exemplo de uso:
'''
csv_files = [
    '../Datasets/landsat_features_more_bands_train.csv',
    '../Datasets/terraclimate_features_more_bands_training.csv',
    # ... adicione todos os seus arquivos aqui
]

# 2. Instanciar o classificador
classifier = FeatureClassifier(csv_training_files=csv_files)

# 3. Definir um modelo rápido para o teste (Random Forest é ótimo para isso)
from sklearn.ensemble import RandomForestRegressor
rf_selector = RandomForestRegressor(n_estimators=50, max_depth=5, n_jobs=-1, random_state=42)

classifier.define_model(rf_selector)

# 4. Rodar a seleção
# Ele vai te retornar um dicionário com as melhores features para cada target
best_features_dict = classifier.classify_features_per_target(feature_to_start='swir22')

# 5. Ver o resultado
print(best_features_dict)

'''
            