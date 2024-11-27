import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from torch_geometric.data import DataLoader
from gnn import GNNModel

model_path = 'gnn_model.pth'  # Path to your saved model
model = GNNModel(num_node_features=..., hidden_dim=..., output_dim=2)  # Define your model architecture here
model.load_state_dict(torch.load(model_path))
model.eval()  # Set model to evaluation mode

# Prepare your test dataset and DataLoader (assumes you have test_loader defined)
# test_loader = DataLoader(...)  # Define the test set DataLoader

test_preds = []
test_labels = []
with torch.no_grad():  # Disable gradient computation during inference
	for batch in test_loader:
		graphs, labels = batch
		outputs = model(graphs)  # Forward pass through the model
		_, predicted = torch.max(outputs, dim=1)  # Get the predicted class (0 or 1)
		test_preds.extend(predicted.cpu().numpy())  # Move predictions to CPU and append to list
		test_labels.extend(labels.cpu().numpy())  # Move true labels to CPU and append to list

# Calculate evaluation metrics
accuracy = accuracy_score(test_labels, test_preds)
precision = precision_score(test_labels, test_preds)
recall = recall_score(test_labels, test_preds)

# Print evaluation results
print(f'Test Accuracy: {accuracy:.4f}')
print(f'Test Precision: {precision:.4f}')
print(f'Test Recall: {recall:.4f}')

# Generate and plot confusion matrix
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
