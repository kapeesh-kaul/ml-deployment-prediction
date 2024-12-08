import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch import optim


# Define the GNN model
class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)  # First GCN layer
        self.conv2 = GCNConv(hidden_dim, output_dim)  # Second GCN layer
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        # x: Node features
        # edge_index: Graph edges
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Model training function
def train_gnn(model, train_graph, device, epochs=100, learning_rate=0.001, batch_size=32):
    train_loader = DataLoader([train_graph], batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    model.train()
    training_log = {}
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)  # Move batch to GPU/CPU
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)  # Forward pass
            loss = criterion(out, batch.y)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update parameters
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        training_log[epoch + 1] = avg_loss
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    
    print("Model training complete.")
    return training_log


# Model evaluation function
def evaluate_gnn(model, test_graph, device, batch_size=32):
    test_loader = DataLoader([test_graph], batch_size=batch_size, shuffle=False)
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)  # Forward pass
            all_predictions.append(out.cpu())
            all_labels.append(batch.y.cpu())

    # Concatenate all predictions and labels
    predictions = torch.cat(all_predictions).view(-1)
    labels = torch.cat(all_labels).view(-1)

    mse = mean_squared_error(labels, predictions)
    r2 = r2_score(labels, predictions)
    return {"Mean Squared Error": mse, "R2 Score": r2}