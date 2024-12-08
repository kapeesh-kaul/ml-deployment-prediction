import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score

# PyTorch model definition
class PolynomialRegressionModel(nn.Module):
    def __init__(self, input_dim, degree):
        super(PolynomialRegressionModel, self).__init__()
        self.degree = degree
        self.output = nn.Linear(input_dim * degree, 1)
    
    def forward(self, x):
        poly_features = torch.cat([x ** (i + 1) for i in range(self.degree)], dim=1)
        return self.output(poly_features)

# Model training function
def train_model(model, X_train, y_train, X_val, y_val, epochs=100, learning_rate=0.001):
    """
    Trains the model while logging training and validation losses.

    Parameters:
        model: The PyTorch model to train.
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer.

    Returns:
        train_losses (dict): Training losses per epoch.
        val_losses (dict): Validation losses per epoch.
    """
    # Initialize model weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    model.apply(init_weights)

    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = {}
    val_losses = {}

    for epoch in range(epochs):
        # Training step
        optimizer.zero_grad()
        train_outputs = model(X_train)
        train_loss = criterion(train_outputs, y_train)
        train_loss.backward()
        optimizer.step()
        
        # Validation step
        model.eval()  # Switch to evaluation mode
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)

        # Log losses for the current epoch
        train_losses[epoch + 1] = train_loss.item()
        val_losses[epoch + 1] = val_loss.item()

        model.train()  # Switch back to training mode

        # Print losses every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

    print("Model training complete.")
    return train_losses, val_losses


# Model evaluation function
def evaluate_model(model, X_test, y_test, y_mean, y_std):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).view(-1) * y_std + y_mean  # Reverse standardization
        y_test = y_test.view(-1) * y_std + y_mean
        mse = mean_squared_error(y_test.cpu(), predictions.cpu())
        r2 = r2_score(y_test.cpu(), predictions.cpu())
    # print(f"Model evaluation complete. MSE: {mse}, R2: {r2}")
    return {"Mean Squared Error": mse, "R2 Score": r2}
