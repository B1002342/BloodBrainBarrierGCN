import torch
from gnn import GNNModel, smiles_to_graph

nicotine_smile = "CN1CCCC1C2=CN(C)C2"

model_path = 'gnn_model.pth'
model = GNNModel(num_node_features=5, hidden_dim=128, output_dim=2)
model.load_state_dict(torch.load(model_path))
model.eval()
nicotine_graph = smiles_to_graph(nicotine_smile)

with torch.no_grad():
	output = model(nicotine_graph) 
	_, predicted = torch.max(output, dim=1)

print(predicted)


