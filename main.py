import torch
import os
import json
import argparse
import mlflow
import mlflow.pytorch
from syn_data import load_dataset
from data_transform.preprocessing import preprocess_data, preprocess_data_to_graph

# Directories for saving parameters
PARAMS_DIR = "model_params"

# Ensure the parameters directory exists
if not os.path.exists(PARAMS_DIR):
    os.makedirs(PARAMS_DIR)


def load_params(model_name):
    """
    Loads model parameters from a JSON file in the PARAMS_DIR directory.
    """
    params_file = os.path.join(PARAMS_DIR, f"{model_name}.json")
    if os.path.exists(params_file):
        with open(params_file, "r") as f:
            params = json.load(f)
        print(f"Loaded parameters from {params_file}")
        return params
    else:
        print(f"No parameter file found for {model_name} in {PARAMS_DIR}. Using empty parameters.")
        return {}


def save_model(model, model_name):
    """
    Saves the model's state_dict to the SAVE_DIR directory.
    """
    save_path = os.path.join("saved_models", f"{model_name}.pth")
    torch.save(model.state_dict(), save_path)  # Save only the model state_dict
    print(f"Model weights saved to {save_path}")


def load_model(model, model_name, device):
    """
    Loads the model's state_dict from the SAVE_DIR directory.
    """
    save_path = os.path.join("saved_models", f"{model_name}.pth")
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))  # Load weights only
        print(f"Model weights loaded from {save_path}")
        return True
    print(f"No saved model found at {save_path}")
    return False


def log_losses_to_mlflow(losses, prefix="loss"):
    """
    Logs a dictionary of losses (epoch -> loss) to MLflow.

    Parameters:
        losses (dict): Dictionary where keys are epochs and values are losses.
        prefix (str): Prefix for the metric name in MLflow.
    """
    for epoch, loss in losses.items():
        mlflow.log_metric(f"{prefix}", loss, step=epoch)


# Polynomial Regression
def run_polynomial_regression(device, overwrite=False, **params):
    """
    Runs the Polynomial Regression model with MLflow tracking.
    """
    with mlflow.start_run(run_name="Polynomial Regression"):
        mlflow.log_params(params)

        df = load_dataset()
        target_column = "total_cost"

        X_train, X_val, X_test, y_train, y_val, y_test, y_mean, y_std = preprocess_data(df, target_column, device=device)

        from models.polynomial_regression import PolynomialRegressionModel, train_model, evaluate_model
        input_dim = X_train.shape[1]
        model = PolynomialRegressionModel(input_dim, params["degree"]).to(device)

        model_name = f"polynomial_regression_deg_{params['degree']}_lr_{params['learning_rate']}"

        if overwrite or not load_model(model, model_name, device):
            train_losses, val_losses = train_model(model, X_train, y_train, X_val, y_val, epochs=params["epochs"], learning_rate=params["learning_rate"])
            
            save_model(model, model_name)

            log_losses_to_mlflow(train_losses, prefix="train_loss")
            log_losses_to_mlflow(val_losses, prefix="val_loss")

        mlflow.pytorch.log_model(model, artifact_path="model")

        train_results = evaluate_model(model, X_train, y_train, y_mean, y_std)
        print("Training Evaluation Results:", train_results)
        mlflow.log_metrics({"train_mse": train_results["Mean Squared Error"], "train_r2": train_results["R2 Score"]})

        test_results = evaluate_model(model, X_test, y_test, y_mean, y_std)
        print("Evaluation Results:", test_results)
        mlflow.log_metrics({"test_mse": test_results["Mean Squared Error"], "test_r2": test_results["R2 Score"]})


# Feedforward Neural Network
def run_feedforward_nn(device, overwrite=False, **params):
    """
    Runs the Feedforward Neural Network model with MLflow tracking.
    """
    with mlflow.start_run(run_name="Feedforward Neural Network"):
        mlflow.log_params(params)

        df = load_dataset()
        target_column = "total_cost"

        X_train, X_val, X_test, y_train, y_val, y_test, y_mean, y_std = preprocess_data(df, target_column, device=device)

        from models.feedforward_neural_network import FeedforwardNeuralNetwork, train_model, evaluate_model
        input_dim = X_train.shape[1]
        model = FeedforwardNeuralNetwork(input_dim, hidden_dim=params["hidden_dim"], num_layers=params["num_layers"]).to(device)

        model_name = f"feedforward_nn_hd_{params['hidden_dim']}_nl_{params['num_layers']}_lr_{params['learning_rate']}"

        if overwrite or not load_model(model, model_name, device):
            train_losses, val_losses = train_model(
                model, 
                X_train, 
                y_train, 
                X_val, 
                y_val, 
                epochs=params["epochs"],
                learning_rate=params["learning_rate"]
            )
            
            save_model(model, model_name)

            # Log to MLflow (if required)
            log_losses_to_mlflow(train_losses, prefix="train_loss")
            log_losses_to_mlflow(val_losses, prefix="val_loss")


        mlflow.pytorch.log_model(model, artifact_path="model")

        train_results = evaluate_model(model, X_train, y_train, y_mean, y_std)
        print("Training Evaluation Results:", train_results)
        mlflow.log_metrics({"train_mse": train_results["Mean Squared Error"], "train_r2": train_results["R2 Score"]})

        test_results = evaluate_model(model, X_test, y_test, y_mean, y_std)
        print("Evaluation Results:", test_results)
        mlflow.log_metrics({"test_mse": test_results["Mean Squared Error"], "test_r2": test_results["R2 Score"]})


# TabTransformer
def run_tab_transformer(device, overwrite=False, **params):
    """
    Runs the TabTransformer model with MLflow tracking.
    """
    with mlflow.start_run(run_name="TabTransformer"):
        mlflow.log_params(params)

        df = load_dataset()
        target_column = "total_cost"

        X_train, X_val, X_test, y_train, y_val, y_test, y_mean, y_std = preprocess_data(df, target_column, device=device)

        from models.tab_transformer import TabTransformer, train_tab_transformer, evaluate_tab_transformer
        input_dim = X_train.shape[1]
        model = TabTransformer(
            input_dim=input_dim,
            embed_dim=params["embed_dim"],
            num_heads=params["num_heads"],
            num_layers=params["num_layers"],
            mlp_hidden_dim=params["mlp_hidden_dim"],
            dropout=params["dropout"]
        ).to(device)

        model_name = f"tab_transformer_ed_{params['embed_dim']}_nh_{params['num_heads']}_nl_{params['num_layers']}_lr_{params['learning_rate']}"

        if overwrite or not load_model(model, model_name, device):
            train_losses, val_losses = train_tab_transformer(
                model, 
                X_train, 
                y_train, 
                X_val, 
                y_val, 
                batch_size=params["batch_size"], 
                epochs=params["epochs"], 
                learning_rate=params["learning_rate"]
            )

            save_model(model, model_name)

            log_losses_to_mlflow(train_losses, prefix="train_loss")
            log_losses_to_mlflow(val_losses, prefix="val_loss")

        mlflow.pytorch.log_model(model, artifact_path="model")

        train_results = evaluate_tab_transformer(model, X_train, y_train, y_mean, y_std)
        print("Training Evaluation Results:", train_results)
        mlflow.log_metrics({"train_mse": train_results["Mean Squared Error"], "train_r2": train_results["R2 Score"]})

        test_results = evaluate_tab_transformer(model, X_test, y_test, y_mean, y_std)
        print("Evaluation Results:", test_results)
        mlflow.log_metrics({"test_mse": test_results["Mean Squared Error"], "test_r2": test_results["R2 Score"]})


# Graph Neural Network (GNN)
def run_gnn(device, overwrite=False, **params):
    """
    Runs the Graph Neural Network (GNN) model with MLflow tracking.
    """
    with mlflow.start_run(run_name="Graph Neural Network"):
        mlflow.log_params(params)

        df = load_dataset()
        target_column = "total_cost"

        train_graph, val_graph, test_graph, y_mean, y_std = preprocess_data_to_graph(
            df, target_column, device=device, k_neighbors=params.get("k_neighbors", 5)
        )

        from models.graph_neural_network import GNN, train_gnn, evaluate_gnn
        input_dim = train_graph.x.shape[1]
        model = GNN(input_dim=input_dim, hidden_dim=params["hidden_dim"], output_dim=params["output_dim"]).to(device)

        model_name = f"gnn_hd_{params['hidden_dim']}_od_{params['output_dim']}_lr_{params['learning_rate']}"

        if overwrite or not load_model(model, model_name, device):
            train_losses, val_losses = train_gnn(
                model=model, 
                train_graph=train_graph, 
                val_graph=val_graph, 
                device=device, 
                epochs=params["epochs"], 
                learning_rate=params["learning_rate"], 
                batch_size=params["batch_size"]
            )
            save_model(model, model_name)

            log_losses_to_mlflow(train_losses, prefix="train_loss")
            log_losses_to_mlflow(val_losses, prefix="val_loss")

        mlflow.pytorch.log_model(model, artifact_path="model")

        train_results = evaluate_gnn(model, train_graph, device)
        print("Training Evaluation Results:", train_results)
        mlflow.log_metrics({"train_mse": train_results["Mean Squared Error"], "train_r2": train_results["R2 Score"]})

        test_results = evaluate_gnn(model, test_graph, device)
        print("Evaluation Results:", test_results)
        mlflow.log_metrics({"test_mse": test_results["Mean Squared Error"], "test_r2": test_results["R2 Score"]})


if __name__ == "__main__":
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("ML Deployment Prediction")

    parser = argparse.ArgumentParser(description="Run different models for prediction")
    parser.add_argument('--model', type=str, required=True, choices=['polynomial_regression', 'feedforward_nn', 'tab_transformer', 'gnn'], help='Model to run')
    parser.add_argument('--overwrite', action='store_true', help='Flag to retrain the model even if a saved version exists')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_params = load_params(args.model)

    if args.model == 'polynomial_regression':
        run_polynomial_regression(device, overwrite=args.overwrite, **model_params)
    elif args.model == 'feedforward_nn':
        run_feedforward_nn(device, overwrite=args.overwrite, **model_params)
    elif args.model == 'tab_transformer':
        run_tab_transformer(device, overwrite=args.overwrite, **model_params)
    elif args.model == 'gnn':
        run_gnn(device=device, overwrite=args.overwrite, **model_params)