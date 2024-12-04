import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
from gnn import GNNModel, smiles_to_graph, MoleculeDataset

df = pd.read_csv('SMILES_data.csv', sep='\t', on_bad_lines='warn')
# df = pd.read_csv('B3DB_usable.csv', on_bad_lines='warn')
test_smiles = df['smiles']
test_labels = df['p_np']

test_graphs = [graph for graph in [smiles_to_graph(smile) for smile in test_smiles] if graph is not None]
test_data = list(zip(test_graphs, test_labels[:len(test_graphs)]))
test_dataset = MoleculeDataset(test_data)
test_loader = GeoDataLoader(test_dataset, batch_size=32, shuffle=False)

test_preds = []
test_labels = []

model_path = 'unbalanced_gnn_model.pth'
# model_path = 'gnn_model.pth'
model = GNNModel(num_node_features=26, hidden_dim=128, output_dim=2)
model.load_state_dict(torch.load(model_path))
model.eval()
with torch.no_grad():
	for batch in test_loader:
		graphs, labels = batch
		outputs = model(graphs)
		_, predicted = torch.max(outputs, dim=1)
		test_preds.extend(predicted.cpu().numpy())
		test_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(test_labels, test_preds)
precision = precision_score(test_labels, test_preds)
recall = recall_score(test_labels, test_preds)

print(f'Test Accuracy: {accuracy:.4f}')
print(f'Test Precision: {precision:.4f}')
print(f'Test Recall: {recall:.4f}')

cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
plt.savefig('SMILES_data_confusion_matrix.png')
