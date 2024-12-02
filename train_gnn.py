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

df = pd.read_csv('B3DB_usable.csv', on_bad_lines='warn')
# df = pd.read_csv('SMILES_data.csv', sep='\t', on_bad_lines='warn')

print(df.columns)

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

smiles_train, smiles_test, labels_train, labels_test = train_test_split(valid_smiles, valid_labels, test_size=0.2, random_state=42, stratify=valid_labels)

train_graphs = [graph for graph in [smiles_to_graph(smile) for smile in smiles_train] if graph is not None]
test_graphs = [graph for graph in [smiles_to_graph(smile) for smile in smiles_test] if graph is not None]
train_data = list(zip(train_graphs, labels_train[:len(train_graphs)]))
test_data = list(zip(test_graphs, labels_test[:len(test_graphs)]))

train_dataset = MoleculeDataset(train_data)
test_dataset = MoleculeDataset(test_data)

train_loader = GeoDataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = GeoDataLoader(test_dataset, batch_size=32, shuffle=False)

print("-" * 30)
print("!!!IGNORE PREVIOUS WARNINGS!!!")
print("-" * 30)

model = GNNModel(num_node_features=26, hidden_dim=128, output_dim=2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

training_losses = []
training_accuracies = []

# Training loop
for epoch in range(200):
	model.train()
	total_loss = 0
	correct_predictions = 0
	total_samples = 0
	for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
		graphs, labels = batch
		optimizer.zero_grad()
		outputs = model(graphs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
		total_loss += loss.item()

		# Calculate accuracy
		_, predicted = torch.max(outputs, dim=1)
		correct_predictions += (predicted == labels).sum().item()
		total_samples += labels.size(0)

	# Average loss and accuracy for the epoch
	epoch_loss = total_loss / len(train_loader)
	epoch_accuracy = correct_predictions / total_samples
	training_losses.append(epoch_loss)
	training_accuracies.append(epoch_accuracy)
	print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy*100:.2f}%')
	
plt.figure(figsize=(8, 6))
plt.plot(range(1, 201), training_losses, label='Training Loss', color='blue')
plt.plot(range(1, 201), training_accuracies, label='Training Accuracy', color='green')
plt.xlabel('Epoch')
# plt.ylabel('Loss')
plt.title('Training Curves')
plt.legend()
plt.grid(True)
plt.savefig('training.png')
plt.show()

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
plt.savefig('B3DB_test_confusion_matrix.png')

# model_path = 'gnn_model.pth'
model_path = 'unbalanced_gnn_model.pth'
torch.save(model.state_dict(), model_path)
print(f'Model saved to {model_path}')