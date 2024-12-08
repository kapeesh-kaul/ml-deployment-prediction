import torch
import matplotlib.pyplot as plt
from syn_data import load_dataset
from data_transform.preprocessing import preprocess_data
import argparse

# Polynomial regression runner
def run_polynomial_regression(device, degree=3, learning_rate=0.001):
    df = load_dataset()
    target_column = "total_cost"

    X_train, X_test, y_train, y_test, y_mean, y_std = preprocess_data(df, target_column, device=device)

    from models.polynomial_regression import PolynomialRegressionModel, train_model, evaluate_model
    input_dim = X_train.shape[1]
    model = PolynomialRegressionModel(input_dim, degree).to(device)

    losses = train_model(model, X_train, y_train, epochs=500, learning_rate=learning_rate)

    train_results = evaluate_model(model, X_train, y_train, y_mean, y_std)
    print("Training Evaluation Results:", train_results)

    results = evaluate_model(model, X_test, y_test, y_mean, y_std)
    print("Evaluation Results:", results)

    return losses, train_results, results

# Feedforward neural network runner
def run_feedforward_nn(device, hidden_dim=64, num_layers=2, learning_rate=0.001):
    df = load_dataset()
    target_column = "total_cost"

    X_train, X_test, y_train, y_test, y_mean, y_std = preprocess_data(df, target_column, device=device)

    from models.feedforward_neural_network import FeedforwardNeuralNetwork, train_model, evaluate_model
    input_dim = X_train.shape[1]
    model = FeedforwardNeuralNetwork(input_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)

    losses = train_model(model, X_train, y_train, epochs=500, learning_rate=learning_rate)

    train_results = evaluate_model(model, X_train, y_train, y_mean, y_std)
    print("Training Evaluation Results:", train_results)

    results = evaluate_model(model, X_test, y_test, y_mean, y_std)
    print("Evaluation Results:", results)

    return losses, train_results, results

# TabTransformer runner
def run_tab_transformer(device, embed_dim=32, num_heads=4, num_layers=2, mlp_hidden_dim=64, dropout=0.1, learning_rate=0.001):
    df = load_dataset()
    target_column = "total_cost"

    X_train, X_test, y_train, y_test, y_mean, y_std = preprocess_data(df, target_column, device=device)

    from models.tab_transformer import TabTransformer, train_tab_transformer, evaluate_tab_transformer
    input_dim = X_train.shape[1]
    model = TabTransformer(
        input_dim=input_dim,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        mlp_hidden_dim=mlp_hidden_dim,
        dropout=dropout,
    ).to(device)

    losses = train_tab_transformer(model, X_train, y_train, epochs=100, learning_rate=learning_rate)

    train_results = evaluate_tab_transformer(model, X_train, y_train, y_mean, y_std)
    print("Training Evaluation Results:", train_results)

    results = evaluate_tab_transformer(model, X_test, y_test, y_mean, y_std)
    print("Evaluation Results:", results)

    return losses, train_results, results

# GNN runner
def run_gnn(device, hidden_dim, output_dim, batch_size=32, learning_rate=0.001, epochs=20):

    df = load_dataset()
    target_column = "total_cost"
    # Preprocess tabular data into graphs
    from data_transform.preprocessing import preprocess_data_to_graph
    train_graph, test_graph, y_mean, y_std = preprocess_data_to_graph(
        df, target_column, test_size=0.2, random_state=42, device=device, k_neighbors=5
    )

    # Initialize GNN model
    from models.graph_neural_network import GNN, train_gnn, evaluate_gnn
    input_dim = train_graph.x.shape[1]
    model = GNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    
    # Train the model
    losses = train_gnn(model, train_graph, device, epochs, learning_rate=learning_rate, batch_size=batch_size)

    # Training results
    train_results = evaluate_gnn(model, train_graph, device)
    print("Training Evaluation Results:", train_results)
    # Evaluate the model
    results = evaluate_gnn(model, test_graph, device)
    print("Evaluation Results:", results)

    return losses, train_results, results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run different models for prediction")
    parser.add_argument('--model', type=str, required=True, choices=['polynomial_regression', 'feedforward_nn', 'tab_transformer', 'gnn'], help='Model to run')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model == 'polynomial_regression':
        run_polynomial_regression(
            device, 
            degree=3,
            learning_rate=0.001
        )
    elif args.model == 'feedforward_nn':
        run_feedforward_nn(
            device, 
            hidden_dim=64, 
            num_layers=2, 
            learning_rate=0.001
        )
    elif args.model == 'tab_transformer':
        run_tab_transformer(
            device,
            embed_dim=32, 
            num_heads=4, 
            num_layers=2, 
            mlp_hidden_dim=64, 
            dropout=0.1,
            learning_rate=0.001
        )
    elif args.model == 'gnn':
        run_gnn(
            device=device,
            hidden_dim=16,
            output_dim=1,
            batch_size=32,
            learning_rate=0.01,
            epochs=100
        )

