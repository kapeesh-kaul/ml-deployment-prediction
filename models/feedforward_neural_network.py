import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score

# PyTorch model definition
class FeedforwardNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(FeedforwardNeuralNetwork, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim if len(layers) == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 1))  # Output layer
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Model training function with backpropagation
def train_model(model, X_train, y_train, X_val, y_val, epochs=100, learning_rate=0.001):
    """
    Trains the model while logging both training and validation losses.

    Parameters:
        model: The PyTorch model to train.
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer.

    Returns:
        training_log (dict): Training losses per epoch.
        validation_log (dict): Validation losses per epoch.
    """
    # Initialize model weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    model.apply(init_weights)

    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    training_log = {}
    validation_log = {}

    for epoch in range(epochs):
        # Training step
        optimizer.zero_grad()  # Reset gradients
        outputs = model(X_train)  # Forward pass
        train_loss = criterion(outputs, y_train)  # Compute loss
        train_loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        # Validation step
        model.eval()  # Switch to evaluation mode
        with torch.no_grad():
            val_outputs = model(X_val)  # Forward pass for validation
            val_loss = criterion(val_outputs, y_val)  # Compute validation loss

        # Log losses
        training_log[epoch + 1] = train_loss.item()
        validation_log[epoch + 1] = val_loss.item()

        # Print losses every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

        model.train()  # Switch back to training mode

    print("Model training complete.")
    return training_log, validation_log

# Model evaluation function
def evaluate_model(model, X_test, y_test, y_mean, y_std):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).view(-1) * y_std + y_mean  # Reverse standardization
        y_test = y_test.view(-1) * y_std + y_mean
        mse = mean_squared_error(y_test.cpu(), predictions.cpu())
        r2 = r2_score(y_test.cpu(), predictions.cpu())
    return {"Mean Squared Error": mse, "R2 Score": r2}