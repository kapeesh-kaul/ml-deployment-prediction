import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score

# Data preprocessing function


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
def train_model(model, X_train, y_train, epochs=100, learning_rate=0.001):
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    model.apply(init_weights)

    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    training_log = {"epoch": [], "loss": []}
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        training_log["epoch"].append(epoch + 1)
        training_log["loss"].append(loss.item())
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
    # print(f"Model evaluation complete. MSE: {mse}, R2: {r2}")
    return {"Mean Squared Error": mse, "R2 Score": r2}
