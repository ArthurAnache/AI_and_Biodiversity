import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# --- Paramètres ---
data_path = "/Users/r/Documents/CS/PROJET_S7/AI_and_Biodiversity/raw_data/#METTRE_LES_2_FICHIERS_ICI"
file = "STOC_pressions_bio_assol_meteo_pesti_routes_lum_120625.csv"
species_name = "APUAPU"  # <- mettre une espèce existante
target_column = "abondance_capped"
hidden_channels = 32
epochs = 50
lr = 0.001
test_size = 0.2

# --- Chargement CSV ---
df = pd.read_csv(f"{data_path}/{file}", sep=";", encoding="latin1", low_memory=False)
print(f"✅ Données chargées ({df.shape[0]} lignes, {df.shape[1]} colonnes)")

# --- Filtrage espèce ---
df_species = df[df['species'] == species_name].copy()
if df_species.empty:
    raise ValueError(f"Aucune donnée pour l'espèce {species_name}")
df_species = df_species.sort_values(by=["id_cell", "Year"])

# --- Création abondance t-1 ---
df_species["abondance_t-1"] = df_species.groupby("id_cell")[target_column].shift(1)
df_species = df_species.dropna(subset=["abondance_t-1"])

# --- Colonnes pressions ---
drop_cols = [c for c in df.columns if 'abondance' in c.lower()] + ['species', 'id_cell', 'Year']
pressure_cols = [c for c in df_species.columns if c not in drop_cols + [target_column]]

# --- Features ---
features_df = df_species[pressure_cols + ["abondance_t-1"]].apply(pd.to_numeric, errors='coerce')

# Remplir NaN par la médiane par colonne
features_df = features_df.fillna(features_df.median())

# Remplacer inf et -inf par la médiane par colonne
for col in features_df.columns:
    median = features_df[col].median()
    features_df[col] = features_df[col].replace([np.inf, -np.inf], median)

# --- Normalisation ---
scaler = StandardScaler()
x_scaled = scaler.fit_transform(features_df.values)

# Remplacer NaN, inf et -inf générés lors de la normalisation
x_scaled = np.nan_to_num(x_scaled, nan=0.0, posinf=0.0, neginf=0.0)

# Conversion en tensor
x = torch.tensor(x_scaled, dtype=torch.float)

# --- Cible ---
y_vals = df_species[target_column].values
y_vals = np.nan_to_num(y_vals, nan=0.0, posinf=0.0, neginf=0.0)
y = torch.tensor(y_vals, dtype=torch.float).unsqueeze(1)

# --- Vérification finale ---
assert not torch.isnan(x).any(), "X contient encore des NaN !"
assert not torch.isnan(y).any(), "y contient encore des NaN !"

# --- Arêtes temporelles t-1 → t ---
edge_index = []
id_cells = df_species['id_cell'].values
years = df_species['Year'].values
for i, (cell, year) in enumerate(zip(id_cells, years)):
    mask = (id_cells == cell) & (years == year - 1)
    if mask.any():
        j = np.where(mask)[0][0]
        edge_index.append([j, i])
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# --- Train/Test split ---
num_nodes = len(y)
indices = np.arange(num_nodes)
split = int(num_nodes * (1 - test_size))
train_idx, test_idx = indices[:split], indices[split:]

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[train_idx] = True
test_mask[test_idx] = True

data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)

# --- Définition GCN ---
class GCNRegression(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# --- Initialisation ---
model = GCNRegression(in_channels=x.shape[1], hidden_channels=hidden_channels)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

# --- Entraînement ---
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    pred = model(data)
    loss = loss_fn(pred[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss={loss.item():.4f}")

# --- Évaluation ---
model.eval()
with torch.no_grad():
    y_pred = model(data)[data.test_mask]
    y_true = data.y[data.test_mask]
    rmse = np.sqrt(mean_squared_error(y_true.numpy(), y_pred.numpy()))
print(f"RMSE sur test set : {rmse:.4f}")

# --- Importance des features par permutation ---
baseline_rmse = rmse
importances = {}
for i, col in enumerate(features_df.columns):
    x_perm = x.clone()
    idx = torch.randperm(num_nodes)
    x_perm[:, i] = x_perm[idx, i]
    data_perm = Data(x=x_perm, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)
    with torch.no_grad():
        y_perm = model(data_perm)[data.test_mask]
        rmse_perm = np.sqrt(mean_squared_error(y_true.numpy(), y_perm.numpy()))
    importances[col] = rmse_perm - baseline_rmse

importances_sorted = sorted(importances.items(), key=lambda x: x[1], reverse=True)
print("\nTop variables importantes :")
for col, imp in importances_sorted[:10]:
    print(f"{col}: {imp:.4f}")

# --- Top 10 variables importantes ---
top_vars = importances_sorted[:10]
names = [v[0] for v in top_vars]
values = [v[1] for v in top_vars]

plt.figure(figsize=(10,6))
plt.barh(names[::-1], values[::-1], color='skyblue', edgecolor='k')  # inversé pour avoir la plus importante en haut
plt.xlabel("Augmentation du RMSE (importance)")
plt.title(f"Top 10 variables importantes pour {species_name}")
plt.tight_layout()
plt.show()