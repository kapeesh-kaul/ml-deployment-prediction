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
def train_model(model, X_train, y_train, epochs=100, learning_rate=0.001):
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    model.apply(init_weights)

    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    training_log = {}

    for epoch in range(epochs):
        optimizer.zero_grad()  # Reset gradients
        outputs = model(X_train)  # Forward pass
        loss = criterion(outputs, y_train)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights)
        
        training_log[epoch + 1] = loss.item()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    print("Model training complete.")
    return training_log

# Model evaluation function
def evaluate_model(model, X_test, y_test, y_mean, y_std):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).view(-1) * y_std + y_mean  # Reverse standardization
        y_test = y_test.view(-1) * y_std + y_mean
        mse = mean_squared_error(y_test.cpu(), predictions.cpu())
        r2 = r2_score(y_test.cpu(), predictions.cpu())
    return {"Mean Squared Error": mse, "R2 Score": r2}