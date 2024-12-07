import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import torch


def preprocess_data(df, target_column, test_size=0.2, random_state=42, device='cpu'):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X.select_dtypes(include=['number']).columns
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(), categorical_cols)
        ]
    )
    
    X = preprocessor.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Standardize the target variable
    y_mean, y_std = y_train.mean(), y_train.std()
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std
    
    # Convert to PyTorch tensors and move to the specified device
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)
    y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(device)
    
    print("Data preprocessing complete. One-hot encoding, normalization, standardization applied adn converted to torch-tensors.")
    return X_train, X_test, y_train, y_test, y_mean, y_std