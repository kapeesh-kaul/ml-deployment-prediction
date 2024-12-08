import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score

# TabTransformer model definition
class TabTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, mlp_hidden_dim, dropout=0.1):
        super(TabTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=mlp_hidden_dim, 
            dropout=dropout, 
            activation="relu", 
            batch_first=True  # Ensure batch_first=True for consistent dimensions
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # Embed the input
        embedded = self.embedding(x)  # (batch_size, embed_dim)
        embedded = embedded.unsqueeze(1)  # (batch_size, seq_length=1, embed_dim)
        # print(f"Transformer Input: {embedded.shape}")

        # Pass through Transformer
        transformed = self.transformer_encoder(embedded)
        # print(f"Transformer Output: {transformed.shape}")  # Should match (batch_size, seq_length=1, embed_dim)

        # Reduce sequence dimension for output layer
        output = self.output_layer(transformed[:, 0, :])  # (batch_size, embed_dim) -> (batch_size, 1)
        # print(f"Output Shape: {output.shape}")

        return output




# Model training function
def train_tab_transformer(model, X_train, y_train, batch_size=1024, epochs=100, learning_rate=0.001):
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    model.apply(init_weights)

    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create DataLoader for mini-batch processing
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    training_log = {}

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)  # Forward pass
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        training_log[epoch + 1] = avg_epoch_loss
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}")

    print("Model training complete.")
    return training_log

# Model evaluation function
from torch.utils.data import DataLoader, TensorDataset

def evaluate_tab_transformer(model, X_test, y_test, y_mean, y_std, batch_size=512):
    model.eval()
    dataset = TensorDataset(X_test, y_test)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_predictions = []
    with torch.no_grad():
        for batch_X, _ in dataloader:
            predictions = model(batch_X)
            all_predictions.append(predictions.cpu())
    
    # Concatenate predictions and reverse standardize
    predictions = torch.cat(all_predictions).view(-1) * y_std + y_mean
    y_test = y_test.view(-1) * y_std + y_mean
    mse = mean_squared_error(y_test.cpu(), predictions.cpu())
    r2 = r2_score(y_test.cpu(), predictions.cpu())
    return {"Mean Squared Error": mse, "R2 Score": r2}
