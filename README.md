# Water Polo Match Prediction

Machine-learning models for **predicting the outcome of water polo matches** from historical match and player statistics. Developed for my B.Sc. thesis *"Predicting the outcome of a water polo match using machine learning"* (University of Ljubljana, Faculty of Computer and Information Science, 2026).

Water polo is largely unexplored in sports analytics compared to other team sports, despite detailed match statistics being available. This project benchmarks 13 approaches — from clustering to graph neural networks — on real match data.

**Best result: 80% prediction accuracy with Random Forest** (116 trees, draws excluded).

## Data

Match and player statistics scraped from [total-waterpolo.com](https://total-waterpolo.com/) with the companion repository [totalwaterpolo-web-scraper](https://github.com/JasaZnidar/totalwaterpolo-web-scraper). Raw JSON event logs are converted into per-player performance vectors; categorical attributes (e.g. dominant hand) are encoded as classification features.

Two classification scenarios are evaluated:
- **With draws** — three classes: home win / away win / draw
- **Without draws** — two classes: home win / away win

## Results

Accuracy on the test set, per model and scenario:

| Model | With draws | Without draws |
|---|---|---|
| **Random Forest** | **76%** | **80%** |
| Gradient Boost | 73% | 78% |
| XGBoost | 75% | 77% |
| Logistic regression | 75% | 75% |
| Linear regression | 71% | 75% |
| Neural network (MLP, PyTorch) | 68% | 75% |
| SVM (linear kernel, C = 3.84) | 69% | 73% |
| Decision tree | 63% | 73% |
| KNN | 61% | 70% |
| Hierarchical clustering | 64% | 69% |
| GNN — homogeneous graph (GCN) | 62% | 67% |
| GNN — heterogeneous graph (Transformer conv) | 65% | 63% |
| K-means clustering | 56% | 56% |

Key findings:
- Tree-based ensemble models performed best, consistent with the literature on tabular data (Grinsztajn et al., 2022).
- Excluding draws simplifies the problem and generally improves accuracy.
- Graph neural networks (homogeneous team graphs and heterogeneous player–team–match graphs) underperformed, most likely due to the limited dataset size — GNNs typically need much larger training sets.

## Models implemented

- **Unsupervised:** K-means, hierarchical clustering (centroid-based prediction)
- **Classical supervised:** KNN, SVM (linear / polynomial / RBF kernels), decision trees, Random Forest, Gradient Boost, XGBoost, linear & logistic regression
- **Deep learning:** multilayer perceptron (PyTorch)
- **Graph neural networks:** homogeneous graphs (GCN, GAT) and heterogeneous player–team–match graphs (TransformerConv, GraphConv, GAT, SAGE)

## Tech stack

Python 3 · scikit-learn · PyTorch (+ torch-geometric for GNNs) · XGBoost · SciPy · NumPy · pandas · Matplotlib

## Usage

```bash
pip install -r requirements.txt

# TODO: adjust to the actual scripts/notebooks in the repo, e.g.:
# python train.py --model random_forest --exclude-draws
```

<!-- TODO: describe repo layout (data prep scripts, per-model scripts/notebooks, where the JSON dataset goes) -->

## Future work

- Larger, more diverse datasets for more robust evaluation
- Transfer to other team sports (football, basketball, handball)
- Richer spatial data: representing players and the ball as graph nodes with interactions as edges, using video-analysis tools (QwikCut, Nacsport) and computer vision

## License

GPL-3.0 — see [LICENSE](LICENSE).

## Author

Jaša Žnidar — B.Sc. thesis, University of Ljubljana, Faculty of Computer and Information Science (mentor: doc. dr. Blaž Meden)
