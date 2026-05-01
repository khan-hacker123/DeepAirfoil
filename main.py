import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# ==========================================
# 1. LOAD AND PREPARE DATA
# ==========================================
print("Loading dataset...")
# Change this to match your exact downloaded CSV filename
df = pd.read_csv(r'C:\programming\AI\airfoil_data.csv')

# Dynamically generate the 62 geometric coefficient column names
upper_coeffs = [f'upperSurfaceCoeff{i}' for i in range(1, 32)]
lower_coeffs = [f'lowerSurfaceCoeff{i}' for i in range(1, 32)]

# Define inputs (64 total) and outputs (2 total)
input_cols = upper_coeffs + lower_coeffs + ['reynoldsNumber', 'alpha']
output_cols = ['coefficientLift', 'coefficientDrag']

# Drop any rows with missing data just to be safe
df = df.dropna(subset=input_cols + output_cols)

X_raw = df[input_cols].values
y_raw = df[output_cols].values

# --- CRITICAL: SCALE THE DATA ---
# Reynolds numbers are huge, coefficients are tiny. We must normalize them.
X_mean = np.mean(X_raw, axis=0)
X_std = np.std(X_raw, axis=0)
# Add small epsilon to prevent division by zero if a column is constant
X_scaled = (X_raw - X_mean) / (X_std + 1e-8) 

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_raw, dtype=torch.float32)

# Train/Test Split (80/20)
train_size = int(0.8 * len(X_tensor))
X_train, y_train = X_tensor[:train_size], y_tensor[:train_size]
X_test, y_test = X_tensor[train_size:], y_tensor[train_size:]

# ==========================================
# 2. NEURAL NETWORK ARCHITECTURE
# ==========================================
class SurrogateCFDModel(nn.Module):
    def __init__(self):
        super(SurrogateCFDModel, self).__init__()
        # 64 inputs -> Hidden layers -> 2 outputs (C_l, C_d)
        self.network = nn.Sequential(
            nn.Linear(64, 128),  # Increased width to handle 64 inputs
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.network(x)

model = SurrogateCFDModel()

# ==========================================
# 3. TRAINING LOOP
# ==========================================
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 500
print(f"Starting Training on {len(X_train)} samples...")

for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Forward pass
    predictions = model(X_train)
    loss = criterion(predictions, y_train)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Training Loss (MSE): {loss.item():.6f}')

# ==========================================
# 4. QUICK TEST EVALUATION
# ==========================================
model.eval()
with torch.no_grad():
    test_predictions = model(X_test)
    test_loss = criterion(test_predictions, y_test)
    print(f"\nFinal Test Loss on unseen data: {test_loss.item():.6f}")
    
    # Show one quick comparison
    print("\n--- Sample Prediction from Test Set ---")
    print(f"Actual C_l, C_d:    {y_test[0].numpy()}")
    print(f"Predicted C_l, C_d: {test_predictions[0].numpy()}")