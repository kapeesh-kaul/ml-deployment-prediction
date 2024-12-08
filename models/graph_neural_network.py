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
def train_gnn(model, train_graph, val_graph, device, epochs=100, learning_rate=0.001, batch_size=32):
    """
    Trains the GNN model and logs training and evaluation (validation) losses.

    Parameters:
        model: The GNN model.
        train_graph: Training graph data.
        val_graph: Validation graph data.
        device: Device to use for training ('cpu' or 'cuda').
        epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer.
        batch_size: Batch size for training.

    Returns:
        train_log (dict): Training losses per epoch.
        val_log (dict): Validation losses per epoch.
    """
    train_loader = DataLoader([train_graph], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader([val_graph], batch_size=batch_size, shuffle=False)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    model.train()
    train_log = {}
    val_log = {}

    for epoch in range(epochs):
        # Training step
        total_train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)  # Move batch to GPU/CPU
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)  # Forward pass
            loss = criterion(out, batch.y)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update parameters
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_log[epoch + 1] = avg_train_loss

        # Validation step
        model.eval()  # Switch to evaluation mode
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index)  # Forward pass
                val_loss = criterion(out, batch.y)  # Compute validation loss
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_log[epoch + 1] = avg_val_loss

        model.train()  # Switch back to training mode

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    print("Model training complete.")
    return train_log, val_log



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