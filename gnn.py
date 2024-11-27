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

# Read the CSV file with correct delimiter and error handling
df = pd.read_csv('SMILES_data.csv', sep='\t', on_bad_lines='warn')

# Check column names
print(df.columns)

# Extract SMILES and labels
smiles_list = df['smiles']
labels = df['p_np']

# Filter out invalid SMILES strings
valid_smiles = []
valid_labels = []
for smile, label in zip(smiles_list, labels):
	try:
		mol = Chem.MolFromSmiles(smile)
		if mol is not None:
			# Sanitize the molecule
			try:
				Chem.SanitizeMol(mol)
			except ValueError as e:
				print(f"Error sanitizing SMILES: {smile} - {e}")
				continue
			valid_smiles.append(smile)
			valid_labels.append(label)
	except Exception as e:
		print(f"Invalid SMILES: {smile} - {e}")

# Split data into training, testing, and validation sets
smiles_train, smiles_test, labels_train, labels_test = train_test_split(valid_smiles, valid_labels, test_size=0.2, random_state=42, stratify=valid_labels)

def smiles_to_graph(smiles):
	try:
		mol = Chem.MolFromSmiles(smiles)
		if mol is None:
			return None
	
		# Sanitize the molecule
		try:
			Chem.SanitizeMol(mol)
		except ValueError as e:
			print(f"Error sanitizing SMILES: {smiles} - {e}")
			return None
	
		atoms = mol.GetAtoms()
		bonds = mol.GetBonds()
	
		atom_features = []
		for atom in atoms:
			feature = [
				atom.GetAtomicNum(),
				atom.GetDegree(),
				atom.GetFormalCharge(),
				atom.GetTotalNumHs(),
				atom.GetNumRadicalElectrons(),
			]
			atom_features.append(feature)
	
		edge_list = []
		edge_features = []
		for bond in bonds:
			edge_list.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
			edge_list.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
	
			# Use GetBondType instead of IsAromatic and IsConjugated
			bond_type = bond.GetBondType()
			feature = [
				bond_type == Chem.BondType.AROMATIC,
				bond_type == Chem.BondType.DOUBLE,
				bond_type == Chem.BondType.SINGLE,
				bond_type == Chem.BondType.TRIPLE,
				bond.IsInRing()
			]
			edge_features.append(feature)
			edge_features.append(feature)
	
		x = torch.tensor(atom_features, dtype=torch.float)
		edge_index = torch.tensor(edge_list, dtype=torch.long).T
		edge_attr = torch.tensor(edge_features, dtype=torch.float)
	
		return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
	except Exception as e:
		print(f"Error processing SMILES: {smiles} - {e}")
		return None

# Convert SMILES to graph data for training, validation, and testing sets
train_graphs = [graph for graph in [smiles_to_graph(smile) for smile in smiles_train] if graph is not None]
# val_graphs = [graph for graph in [smiles_to_graph(smile) for smile in smiles_val] if graph is not None]
test_graphs = [graph for graph in [smiles_to_graph(smile) for smile in smiles_test] if graph is not None]
# print(test_graph)

# Combine with labels
train_data = list(zip(train_graphs, labels_train[:len(train_graphs)]))
test_data = list(zip(test_graphs, labels_test[:len(test_graphs)]))

class MoleculeDataset(Dataset):
	def __init__(self, data):
		self.data = data

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		graph, label = self.data[idx]
		return graph, label

# Create datasets and data loaders
train_dataset = MoleculeDataset(train_data)
test_dataset = MoleculeDataset(test_data)

train_loader = GeoDataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = GeoDataLoader(test_dataset, batch_size=32, shuffle=False)

print("-" * 30)
print("!!!IGNORE PREVIOUS WARNINGS!!!")
print("-" * 30)

class GNNModel(nn.Module):
	def __init__(self, num_node_features, hidden_dim, output_dim):
		super(GNNModel, self).__init__()
		self.conv1 = GCNConv(num_node_features, hidden_dim)
		self.conv2 = GCNConv(hidden_dim, hidden_dim)
		self.fc = nn.Linear(hidden_dim, output_dim)
	
	def forward(self, data):
		x, edge_index, batch = data.x, data.edge_index, data.batch
	
		x = self.conv1(x, edge_index)
		x = F.relu(x)
		x = F.dropout(x, training=self.training)
		x = self.conv2(x, edge_index)
		x = F.relu(x)
		x = F.dropout(x, training=self.training)
	
		# Use global mean pooling to get a fixed-size output per graph
		x = global_mean_pool(x, batch)
	
		x = self.fc(x)
	
		return x

# Initialize the GNN model
model = GNNModel(num_node_features=5, hidden_dim=128, output_dim=2) # Assuming binary classification

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(200):
	model.train()
	total_loss = 0
	for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
		graphs, labels = batch
		optimizer.zero_grad()
		outputs = model(graphs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
		total_loss += loss.item()
	print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}')

# Evaluate the model on the test set
model.eval()
test_preds = []
test_labels = []
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

# Plot confusion matrix using seaborn heatmap
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

model_path = 'gnn_model.pth'  # Choose the path where to save the model
torch.save(model.state_dict(), model_path)
print(f'Model saved to {model_path}')
