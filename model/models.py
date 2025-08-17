import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd


class StockCNNOneHot(nn.Module):
    """
    1D-CNN model that concatenates a one-hot stock identity vector to the
    per-timestep feature vector, then applies a temporal convolution
    and a global pooling head to regress next-day return.
    """
    def __init__(self, input_dim, stock_num, conv_out_channels=32, kernel_size=3):
        super().__init__()
        self.stock_num = stock_num
        self.input_dim = input_dim
        # Effective input channels to Conv1d after concatenating one-hot stock id
        self.conv_input_dim = input_dim + stock_num

        # Conv1D expects input shape: (batch_size, channels, seq_len)
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_input_dim,
            out_channels=conv_out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,  # "same" padding to preserve seq_len
        )

        # Head: temporal global pooling -> flatten -> linear regression
        self.output_layer = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # compress time dimension to length 1
            nn.Flatten(),             # (batch_size, out_channels)
            nn.Linear(conv_out_channels, 1),  # final regression layer
        )

    def forward(self, x_seq, stock_id):
        """
        Args:
            x_seq: Tensor of shape (batch_size, seq_len, input_dim)
            stock_id: Tensor of shape (batch_size,) with integer stock ids
        Returns:
            Tensor of shape (batch_size, 1): predicted return
        """
        batch_size, seq_len, _ = x_seq.shape

        # One-hot encode stock ids: (batch_size, stock_num)
        stock_one_hot = F.one_hot(stock_id, num_classes=self.stock_num).float()

        # Repeat the one-hot vector across all timesteps: (batch_size, seq_len, stock_num)
        stock_feat_seq = stock_one_hot.unsqueeze(1).repeat(1, seq_len, 1)

        # Concatenate along the feature dimension: (batch, seq_len, input_dim + stock_num)
        x_combined = torch.cat([x_seq, stock_feat_seq], dim=-1)

        # Rearrange to Conv1d format: (batch, channels, seq_len)
        x_conv = x_combined.permute(0, 2, 1)

        # Temporal convolution
        x_feat = self.conv1d(x_conv)  # (batch, out_channels, seq_len)

        # Aggregate across time and regress
        out = self.output_layer(x_feat)  # (batch, 1)
        return out


class StockLSTM(nn.Module):
    """
    LSTM model that augments each timestep with a learned stock embedding
    and uses the last hidden state for regression.
    """
    def __init__(self, input_dim, hidden_dim, stock_num, emb_dim, num_layers=2):
        super().__init__()
        self.stock_emb = nn.Embedding(stock_num, emb_dim)

        # Concatenated input dimension = original features + stock embedding
        lstm_input_dim = input_dim + emb_dim

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )

        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x_seq, stock_id):
        """
        Args:
            x_seq: (batch, seq_len, input_dim)
            stock_id: (batch,) integer stock ids
        Returns:
            (batch, 1) predicted return
        """
        # Look up stock embedding: (batch, emb_dim)
        stock_vec = self.stock_emb(stock_id)

        # Broadcast embedding to every timestep (repeat to seq_len)
        seq_len = x_seq.size(1)
        stock_vec_expanded = stock_vec.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, emb_dim)

        # Concatenate embedding with the input sequence
        lstm_input = torch.cat([x_seq, stock_vec_expanded], dim=2)  # (batch, seq_len, input_dim + emb_dim)

        # Run LSTM
        _, (h_n, _) = self.lstm(lstm_input)
        h_last = h_n[-1]  # last layer's hidden state

        return self.output_layer(h_last)


class StockTransformer(nn.Module):
    """
    Transformer encoder model that concatenates a learned stock embedding
    to each timestep, projects to d_model, adds learnable positional encodings,
    and uses the last timestep representation for regression.
    """
    def __init__(self, input_dim, stock_num, emb_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.stock_emb = nn.Embedding(stock_num, emb_dim)

        # Concatenated features are projected to Transformer d_model
        self.input_proj = nn.Linear(input_dim + emb_dim, d_model)

        # Simple learnable positional encoding (max sequence length = 500)
        self.pos_encoding = nn.Parameter(torch.randn(1, 500, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final regression head
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, x_seq, stock_id):
        """
        Args:
            x_seq: (batch, seq_len, input_dim)
            stock_id: (batch,) integer stock ids
        Returns:
            (batch, 1) predicted return
        """
        batch_size, seq_len, _ = x_seq.size()

        # Expand stock embedding to each timestep
        stock_vec = self.stock_emb(stock_id)                    # (batch, emb_dim)
        stock_vec_expanded = stock_vec.unsqueeze(1).expand(-1, seq_len, -1)

        # Concatenate embedding to input features
        x = torch.cat([x_seq, stock_vec_expanded], dim=2)       # (batch, seq_len, input_dim + emb_dim)

        # Project to d_model
        x = self.input_proj(x)                                  # (batch, seq_len, d_model)

        # Add positional encoding (learnable)
        x = x + self.pos_encoding[:, :seq_len, :]

        # Transformer encoder
        x_transformed = self.transformer(x)                     # (batch, seq_len, d_model)

        # Use the last timestep representation
        x_last = x_transformed[:, -1, :]                        # (batch, d_model)

        return self.output_layer(x_last)                        # (batch, 1)


def train_model(model, train_loader, val_loader, epochs=20):
    """
    Generic training loop with MSE loss and Adam optimizer.
    Trains on train_loader and reports validation loss on val_loader.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X_batch, stock_batch, y_batch in train_loader:
            X_batch, y_batch, stock_batch = (
                X_batch.to(device),
                y_batch.to(device),
                stock_batch.to(device),
            )
            pred = model(X_batch, stock_batch)
            loss = criterion(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, stock_batch, y_batch in val_loader:
                X_batch, y_batch, stock_batch = (
                    X_batch.to(device),
                    y_batch.to(device),
                    stock_batch.to(device),
                )
                pred = model(X_batch, stock_batch)
                loss = criterion(pred, y_batch)
                val_loss += loss.item()

        # Uncomment if you want to log:
        # print(f"[{epoch+1}] Train Loss: {total_loss / len(train_loader):.4f}  "
        #       f"Val Loss: {val_loss / len(val_loader):.4f}")


# Selected features for modeling
features = [
    "close",         # Closing price (basic price reference)
    "atr_14",        # 14-day Average True Range (short-term volatility)
    "macd_hist",     # MACD histogram (momentum change strength)
    "volume_sma_20", # 20-day SMA of volume (smoothed volume trend)
    "rsi_14",        # 14-day RSI (momentum / overbought-oversold)
    "ma_20",         # 20-day moving average (trend indicator)
    "bb_middle",     # Middle band of Bollinger Bands (trend baseline)
]


def predict_returns(model, df, window=30):
    """
    Slice rolling windows for each stock, run the model in inference mode,
    and return a DataFrame of predicted returns.

    Args:
        model: trained PyTorch model with forward(x_seq, stock_id)
        df (pd.DataFrame): must include columns ['date','stock', features..., 'daily_sentiment']
        window (int): input sequence length (number of past days)
    Returns:
        pd.DataFrame with columns ['date','stock','pred_return']
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    X_list, stock_list, date_list = [], [], []

    # Ensure deterministic order
    df = df.sort_values(["date", "stock"])
    # Map stock symbol -> integer id
    stock_ids = {s: i for i, s in enumerate(df["stock"].unique())}

    # Build rolling windows per stock
    for stock in df["stock"].unique():
        df_stock = df[df["stock"] == stock]
        for i in range(len(df_stock) - window):
            x_seq = df_stock.iloc[i : i + window][features + ["daily_sentiment"]].values
            date = df_stock.iloc[i + window]["date"]
            X_list.append(x_seq)
            stock_list.append(stock_ids[stock])
            date_list.append(date)

    # Convert to tensors
    X_tensor = torch.tensor(np.array(X_list), dtype=torch.float32).to(device)
    stock_tensor = torch.tensor(np.array(stock_list), dtype=torch.long).to(device)

    # Inference
    with torch.no_grad():
        preds = model(X_tensor, stock_tensor).squeeze().cpu().numpy()

    # Collect results
    result_df = pd.DataFrame(
        {
            "date": date_list,
            "stock": [list(stock_ids.keys())[i] for i in stock_list],
            "pred_return": preds,
        }
    )

    return result_df