import torch
from syn_data import load_dataset
from data_transform.preprocessing import preprocess_data

def run_polynomial_regression(device, degree=3):
    df = load_dataset()
    target_column = "total_cost"

    X_train, X_test, y_train, y_test, y_mean, y_std = preprocess_data(df, target_column, device=device)

    from models.polynomial_regression import PolynomialRegressionModel, train_model, evaluate_model
    input_dim = X_train.shape[1]
    model = PolynomialRegressionModel(input_dim, degree).to(device)

    losses = train_model(model, X_train, y_train, epochs=500)

    results = evaluate_model(model, X_test, y_test, y_mean, y_std)
    print("Evaluation Results:", results)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_polynomial_regression(device, degree=3)

