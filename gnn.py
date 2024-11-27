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

class MoleculeDataset(Dataset):
	def __init__(self, data):
		self.data = data

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		graph, label = self.data[idx]
		return graph, label

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