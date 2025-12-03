# Tutorial: Node Classification and Link Prediction in Citation Networks using Graph Neural Networks

*December 2025 · 15 min read · [Open in Colab](https://colab.research.google.com/github/BogGoro/predicting-paper-topics-and-connections-tutorial/blob/main/GraphML_Tutorial.ipynb)*

## Introduction

Graph Machine Learning (Graph ML) is revolutionizing how we analyze interconnected data. From social networks to molecular structures and citation networks, graphs are everywhere. In this tutorial, we'll dive deep into applying Graph Neural Networks (GNNs) to a classic problem: **analyzing academic citation networks**.

We'll build a system that can:
1. **Classify research papers** into topics
2. **Predict potential citations** between papers

All using the powerful **PyTorch Geometric** library and real-world data from the Cora dataset.

## Why This Matters

Citation networks present unique challenges:
- **Relational data**: Papers influence each other through citations
- **Homophily**: Similar papers tend to cite each other
- **Semi-supervised setting**: We have labels for only some papers

Traditional ML struggles with graph data because it ignores relationships between data points. GNNs solve this by learning from both node features **and** graph structure.

## The Cora Dataset

```python
from torch_geometric.datasets import Planetoid

dataset = Planetoid(name='Cora')
data = dataset[0]

print(f"Nodes: {data.num_nodes}")      # 2,708 papers
print(f"Edges: {data.num_edges}")      # 10,556 citations
print(f"Features: {data.num_features}") # 1,433-dimensional word vectors
print(f"Classes: {dataset.num_classes}") # 7 research topics
```

![Class Distribution](images/class_distribution.png?raw=true)

# Graph Neural Networks

## The Magic of Message Passing

At the heart of every GNN is the message passing paradigm:

```text
For each node:
    1. Gather messages from neighbors
    2. Aggregate messages (sum, mean, max)
    3. Update node representation
```

This simple but powerful idea allows information to propagate through the graph, enabling nodes to learn from their local neighborhoods.

## Why GraphSAGE?

We chose GraphSAGE (SAmple and aggreGatE) for several reasons:

- Inductive learning: Can handle new, unseen nodes
- Scalability: Samples neighborhoods instead of using all neighbors
- Flexibility: Multiple aggregation functions available

# Model Architecture

Our joint model consists of three main components:

```python
class JointGraphModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        # Shared GraphSAGE encoder
        self.encoder = GraphSAGEEncoder(in_channels, hidden_channels, hidden_channels)
        
        # Task-specific heads
        self.node_classifier = nn.Linear(hidden_channels, num_classes)
        self.link_predictor = LinkPredictor(hidden_channels, hidden_channels // 2)
```

## Training Strategy

We use a **two-phase training** approach:

**Phase 1**: Train for node classification using labeled papers \
**Phase 2**: Train for link prediction using learned embeddings \
**Optional**: Joint fine-tuning of both tasks

# Implementation Details

## Data Preparation for Link Prediction

Link prediction requires careful handling of edges:

```python
def prepare_link_prediction_data(data, val_ratio=0.1, test_ratio=0.1):
    """Split edges into train/val/test sets"""
    # Make edges undirected and unique
    row, col = data.edge_index
    mask = row < col
    row, col = row[mask], col[mask]
    
    # Random split
    n_val = int(n_edges * val_ratio)
    n_test = int(n_edges * test_ratio)
    
    # Create positive examples
    val_edges_pos = torch.stack([row[:n_val], col[:n_val]], dim=0)
    # ... and negative examples using negative_sampling()
```

## The Training Loop

```python
# Node classification training
for epoch in range(epochs):
    model.train()
    _, node_logits = model(data.x, data.edge_index)
    loss = criterion(node_logits[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
```

# Results

![Class Distribution](images/training_results_comprehensive.png?raw=true)

# Model Interpretation

## t-SNE Visualization

By visualizing learned embeddings, we can see how the model clusters papers by topic:

```python
# Apply t-SNE to embeddings
tsne = TSNE(n_components=2, perplexity=30)
embeddings_2d = tsne.fit_transform(embeddings)

# Plot colored by research topic
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
            c=labels, cmap='tab20', alpha=0.6)
```

![Class Distribution](images/tsne_embeddings.png?raw=true)

The clear separation between clusters shows that our model has learned meaningful representations!

# Practical Application: Citation Recommendation

One exciting application is recommending citations for researchers:

```python
def recommend_citations(model, data, paper_id, top_k=5):
    """Find similar papers that might be worth citing"""
    # Get paper embedding
    paper_embedding = model.get_embeddings(data.x, data.edge_index)[paper_id]
    
    # Find most similar papers (excluding existing citations)
    similarities = cosine_similarity(paper_embedding, all_embeddings)
    
    return top_k most similar papers
```

Example output:
```
Citation Recommendations for Paper 42 (Neural Networks):

1. Paper 128 (Neural Networks) - Similarity: 0.892
2. Paper 256 (Neural Networks) - Similarity: 0.876  
3. Paper 512 (Probabilistic Methods) - Similarity: 0.743
```
