# BloodBrainBarrierGCN

In this project, we train a Graph Neural Network (GNN) on molecular data represented as SMILES
(Simplified Molecular Input Line Entry System) strings, specifically focusing on the structural
features that determine Blood-Brain Barrier (BBB) permeability. By analyzing a curated dataset
of nicotine derivatives, our model provides insight into the molecular features that influence BBB
penetration, offering a stepping stone toward designing safer nicotine analogs. The goal of this work
is not merely to predict BBB permeability but to interpret the underlying molecular patterns that
differentiate compounds that cross the barrier from those that do not.

We exploit the graph structure of biological molecules and accordingly use a GNN for classificiation. Our model is trained on the [Blood-Brain Barrier Database (B3DB)](https://github.com/theochem/B3DB?tab=readme-ov-file). We additionally scrape 172 CAS (Chemical Abstract Service) numbers different nicotine derivatives from [the list of Cymitquimica’s nicotine analog](https://cymitquimica.com/categories/1828/?srsltid=AfmBOoq192jP0XqDRimxlFKcuX8rhkztsdPnmg0H4XYFqxYG5MMRi9ij) and generate SMILES for 164 of them from the [PubChem molecule database](https://pubchem.ncbi.nlm.nih.gov). To the best of our knowledge, there is no other dataset of this type available, and there is ample room for growth. This data, along with BBB permeability predictions made by our GNN model can be found [here](nicotine_derivatives_with_predictions.csv)
