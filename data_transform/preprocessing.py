import torch
from torch_geometric.data import Data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph
import numpy as np



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

# Preprocessing function to convert tabular data to graphs
def preprocess_data_to_graph(df, target_column, test_size=0.2, random_state=42, device='cpu', k_neighbors=5):
    """
    Preprocess tabular data and convert it into graph data for GNNs.
    
    Parameters:
        df (pd.DataFrame): The input dataset.
        target_column (str): The target column name.
        test_size (float): Fraction of the data for testing.
        random_state (int): Random seed for train-test split.
        device (str): Device to move the data to ('cpu' or 'cuda').
        k_neighbors (int): Number of neighbors for graph construction.

    Returns:
        train_graph (Data): Graph data object for training.
        test_graph (Data): Graph data object for testing.
        y_mean (float): Mean of the target variable (training).
        y_std (float): Standard deviation of the target variable (training).
    """
    # Split features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Preprocessing: scale numeric features, one-hot encode categorical features
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X.select_dtypes(include=['number']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(), categorical_cols)
        ]
    )

    X = preprocessor.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Standardize target variable
    y_mean, y_std = y_train.mean(), y_train.std()
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    # Create graph structure using k-nearest neighbors
    def create_graph(X, y):
        adj_matrix = kneighbors_graph(X, n_neighbors=k_neighbors, mode='connectivity', include_self=False)
        edge_index = torch.tensor(np.array(adj_matrix.nonzero()), dtype=torch.long)  # (2, num_edges)
        x_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1).to(device)
        graph = Data(x=x_tensor, edge_index=edge_index, y=y_tensor)
        return graph

    # Create training and testing graph data
    train_graph = create_graph(X_train, y_train)
    test_graph = create_graph(X_test, y_test)

    print("Data preprocessing complete. Converted to graph format.")
    return train_graph, test_graph, y_mean, y_std