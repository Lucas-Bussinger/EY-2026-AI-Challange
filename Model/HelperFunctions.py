# Feature preprocessing and data splitting
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.spatial import cKDTree

# Machine Learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from datetime import date
from tqdm import tqdm
import os 


# Combine two datasets vertically (along columns) using pandas concat function.
def combine_two_datasets(dataset1,dataset2,dataset3):
    '''
    Returns a  vertically concatenated dataset.
    Attributes:
    dataset1 - Dataset 1 to be combined 
    dataset2 - Dataset 2 to be combined
    '''
    
    data = pd.concat([dataset1,dataset2,dataset3], axis=1)
    data = data.loc[:, ~data.columns.duplicated()]
    return data

def split_data(X, y, test_size=0.3, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def evaluate_model(model, X_scaled, y_true, dataset_name="Test"):
    y_pred = model.predict(X_scaled)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\n{dataset_name} Evaluation:")
    print(f"R²: {r2:.3f}")
    print(f"RMSE: {rmse:.3f}")
    return y_pred, r2, rmse




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