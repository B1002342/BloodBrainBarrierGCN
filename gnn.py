import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski
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
		
		functional_groups = {
			"alcohol": Chem.MolFromSmarts("[CX4][OH]"),
			"aldehyde": Chem.MolFromSmarts("[CX3H1](=O)[#6]"),
			"ketone": Chem.MolFromSmarts("[CX3](=O)[#6]"),
			"amine": Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O)]"),
			"carboxylic_acid": Chem.MolFromSmarts("C(=O)[OH]"),
			"aromatic_ring": Chem.MolFromSmarts("a"),
			"ether": Chem.MolFromSmarts("[CX4][OX2][CX4]"),
			"amide": Chem.MolFromSmarts("[NX3][CX3](=O)[#6]"),
			"halogen": Chem.MolFromSmarts("[F,Cl,Br,I]"),
		}
		fg_counts = {fg: len(mol.GetSubstructMatches(pattern)) for fg, pattern in functional_groups.items()}
	
		atom_features = []
		for atom in atoms:
			feature = [
				atom.GetAtomicNum(),                         					# Atomic number
				atom.GetDegree(),                            					# Degree
				atom.GetFormalCharge(),                      					# Formal charge
				atom.GetTotalNumHs(),                        					# Total hydrogens
				atom.GetNumRadicalElectrons(),               					# Radical electrons
				atom.GetIsAromatic(),                        					# Aromaticity
				Chem.Crippen.MolLogP(mol),                   					# LogP (lipophilicity)
				Chem.rdMolDescriptors.CalcTPSA(mol),         					# Topological polar surface area
				Chem.Lipinski.NumHDonors(mol),              					# Number of hydrogen bond donors
				Chem.Lipinski.NumHAcceptors(mol),           					# Number of hydrogen bond acceptors
				Chem.rdMolDescriptors.CalcNumRings(mol),    					# Number of rings
				Chem.rdMolDescriptors.CalcExactMolWt(mol),   					# Molecular weight
				Chem.Lipinski.NumRotatableBonds(mol),		 					# Number of rotatable bonds
				Chem.rdMolDescriptors.CalcNumRings(mol),						# Number of ring structures
				atom.GetHybridization() == Chem.rdchem.HybridizationType.SP,   	# SP hybridization
				atom.GetHybridization() == Chem.rdchem.HybridizationType.SP2,  	# SP2 hybridization
				atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3,  	# SP3 hybridization
				fg_counts["alcohol"],                        # Alcohol count
				fg_counts["aldehyde"],                       # Aldehyde count
				fg_counts["ketone"],                         # Ketone count
				fg_counts["amine"],                          # Amine count
				fg_counts["carboxylic_acid"],                # Carboxylic acid count
				fg_counts["aromatic_ring"],                  # Aromatic ring count
				fg_counts["ether"],                          # Ether count
				fg_counts["amide"],                          # Amide count
				fg_counts["halogen"],                        # Halogen count
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
				bond.IsInRing(),
				bond.GetIsConjugated(),
				bond.GetStereo()
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