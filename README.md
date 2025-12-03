# Tutorial: Node Classification and Link Prediction in Citation Networks using Graph Neural Networks

*December 2025 · 15 min read · [Open in Colab](https://colab.research.google.com/github/BogGoro/predicting-paper-topics-and-connections-tutorial/blob/main/GraphML_Tutorial.ipynb)*

# Introduction

Graph Machine Learning (Graph ML) is revolutionizing how we analyze interconnected data. From social networks to molecular structures and citation networks, graphs are everywhere. In this tutorial, we'll dive deep into applying Graph Neural Networks (GNNs) to a classic problem: **analyzing academic citation networks**.

We'll build a system that can:

1. **Classify research papers** into topics
2. **Predict potential citations** between papers

All using the powerful **PyTorch Geometric** library and real-world data from the Cora dataset.

# Why This Matters

Citation networks present unique challenges:

* **Relational data**: Papers influence each other through citations
* **Homophily**: Similar papers tend to cite each other
* **Semi-supervised setting**: We have labels for only some papers

Traditional ML struggles with graph data because it ignores relationships between data points. GNNs solve this by learning from both node features **and** graph structure.

## The Cora Dataset

The Cora dataset is the "MNIST" of Graph ML - a standard benchmark that every researcher uses. It consists of:

- 2,708 academic papers from 7 machine learning subfields
- 10,556 citation links (directed, making a citation graph)
- 1,433-dimensional binary word vectors (bag-of-words representation)
- 7 research topics: Case-Based, Genetic Algorithms, Neural Networks, Probabilistic Methods, Reinforcement Learning, Rule Learning, Theory

What makes Cora interesting for GNN research:

- It exhibits homophily: Papers cite others in the same research area
- It's sparse: Most papers only cite 3-4 other papers

![Class Distribution](images/class_distribution.png?raw=true)

# Graph Neural Networks

## The Magic of Message Passing

At the heart of every GNN is the message passing paradigm:

For each node:
1. Gather messages from neighbors
2. Aggregate messages (sum, mean, max)
3. Update node representation

This simple but powerful idea allows information to propagate through the graph, enabling nodes to learn from their local neighborhoods.

## Graph Data Structure in PyG

The fundamental shift from traditional ML to Graph ML starts with data representation:

```python
# In PyG, graphs are Data objects with two key components:
# 1. Node features (like traditional features)
# 2. Edge index (the graph structure)

from torch_geometric.data import Data

# Edge index defines relationships: [source_nodes, target_nodes]
# Each column represents one edge
edge_index = torch.tensor([[0, 1, 1, 2],   # Source nodes
                           [1, 2, 0, 1]],  # Target nodes
                          dtype=torch.long)

# Node features: [num_nodes, num_features]
x = torch.randn(3, 16)  # 3 nodes, 16 features each

# Combine into graph
data = Data(x=x, edge_index=edge_index)
```

**Key Insight**: Traditional ML treats each paper as independent. Graph ML understands that papers exist in a citation network where relationships matter. A paper about "Neural Networks" citing another "Neural Networks" paper gives us information that pure text analysis would miss.

## Why GraphSAGE?

We chose GraphSAGE (SAmple and aggreGatE) for several reasons:

* Inductive learning: Can handle new, unseen nodes
* Scalability: Samples neighborhoods instead of using all neighbors
* Flexibility: Multiple aggregation functions available

### GraphSAGE's Neighborhood Sampling

```python
# The key advantage over traditional GCN:
# GraphSAGE samples k neighbors instead of using ALL neighbors

from torch_geometric.nn import SAGEConv

class GraphSAGELayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # This single layer does:
        # 1. Sample fixed-size neighborhood per node
        # 2. Aggregate sampled neighbors' features
        # 3. Update node representation
        self.conv = SAGEConv(in_channels, out_channels)
    
    def forward(self, x, edge_index):
        # Input: x=[2708, 1433], edge_index=[2, 10556]
        # Output: x=[2708, 128] - enriched with neighbor info
        return self.conv(x, edge_index)
```
Mathematically, each typical GraphSAGE layer computes:

**1. Neighbor Sampling**
$N(v) = \text{SampleNeighbors}(v)$

**2. Aggregation** (example: mean aggregator)
$\mathbf{h}{N(v)}^{(k)} = \frac{1}{|N(v)|} \sum{u \in N(v)} \mathbf{h}_u^{(k-1)}$

**3. Node Update**
$\mathbf{h}_v^{(k)} = \sigma\left( W^{(k)} \cdot [\mathbf{h}v^{(k-1)} \Vert \mathbf{h}{N(v)}^{(k)}] \right)$

where:

* $( \mathbf{h}_v^{(k)} )$ — embedding of node (v) at layer (k)
* $(N(v))$ — sampled neighbors
* $(\Vert)$ — concatenation
* $(\sigma)$ — non-linearity (ReLU)

### Aggregator Variants

* **Mean**: smooths neighborhood features
* **LSTM aggregator**: learns sequence-dependent patterns
* **Pool aggregator**: applies MLP + max-pool for expressive power

Key Difference from GCN:

- GCN: Uses full adjacency matrix → O(N²) memory, can't handle new papers
- GraphSAGE: Samples k neighbors per node → O(N·k) memory, works on new graphs

For Cora, this means we can:

1. Train on the 2,708 papers
2. Later add new papers and predict their topics
3. Recommend citations for papers we've never seen before

# Multi-Task Architecture

## The Problem with Single Encoder

```python
# it causes task interference
class NaiveModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_encoder = SAGEConv(...)  # One encoder for both tasks
        self.classifier = Linear(...)        # For node classification
        self.predictor = Linear(...)         # For link prediction
```

**Why this fails**: Node classification needs features that separate classes, while link prediction needs features that capture similarity between nodes. One encoder can't optimize for both.

## Our Solution

```python
class DualEncoderModel(nn.Module):
    def __init__(self):
        super().__init__()
        # SEPARATE encoders for separate tasks
        self.node_encoder = SAGEConv(...)    # Optimized for classification
        self.link_encoder = SAGEConv(...)    # Optimized for similarity
    
    def forward_node(self, x, edge_index):
        # Path 1: Classification-focused embeddings
        z_node = self.node_encoder(x, edge_index)
        return self.node_classifier(z_node)  # Predict topics
    
    def forward_link(self, x, edge_index):
        # Path 2: Similarity-focused embeddings  
        z_link = self.link_encoder(x, edge_index)
        return self.link_predictor(z_link)   # Predict citations
```

Why Two Encoders Work:

- Node classification encoder: Learns to separate "Neural Networks" from "Genetic Algorithms"
- Link prediction encoder: Learns that similar papers (in embedding space) should cite each other
- No task interference: Each encoder specializes for its objective

# Link Prediction Challenges

## The Negative Sampling Problem

Link prediction is fundamentally different from node classification:

```python
from torch_geometric.utils import negative_sampling

# Positive edges: Real citations from Cora
pos_edges = data.edge_index  # [2, 10556] - actual citations

# Generate negative edges: Random non-citations
neg_edges = negative_sampling(
    edge_index=data.edge_index,
    num_nodes=data.num_nodes,           # 2708 papers
    num_neg_samples=pos_edges.size(1)   # Same as positives
)

# Now we have balanced binary classification:
# Positive class: 10,556 real citations
# Negative class: 10,556 fake citations
```

**The Challenge**: For Cora's 2,708 papers, there are potentially 2,708 × 2,707 = 7.3 million possible citations. We only observe 10,556 real ones. The model must learn to distinguish real citations from the 7.3 million possible fake ones.

## Why This is Hard

1. Extreme class imbalance: 10,556 positives vs 7.3 million possible negatives
2. Missing data problem: We don't know if unobserved edges are "won't cite" or "haven't cited yet"
3. Transductive setting: Can't easily test on papers unseen during training

Our solution: Train on a subset of negatives (via sampling) and evaluate ranking quality rather than binary accuracy.

# Evaluation Metrics

## For Node Classification: Standard Accuracy

Since Cora has 7 balanced test classes (20% of papers), we use standard accuracy:

## For Link Prediction: Ranking Quality Matters

We use three complementary metrics:

### 1. ROC-AUC (Area Under ROC Curve)

Measures overall ability to distinguish real vs fake citations.

- AUC = 0.5: Random guessing
- AUC = 1.0: Perfect discrimination
- Interpretation: Probability that a random real citation scores higher than a random fake citation

### 2. Average Precision (AP)

Better than AUC when positives are rare (like citations).

- Rewards models that rank true citations highly
- More relevant for recommendation systems
- Formula: $\text{AP} = \sum_n (R_n - R_{n-1}) P_n$

### 3. Precision@100

Practical metric for real-world systems.

- If we recommend top 100 citations, how many are correct?
- Users only see top recommendations, not full ranking
- $\text{Precision@100} = \frac{\text{number of correct in top 100}}{100}$

**Why multiple metrics?** AUC tells us about global ranking, AP about early ranking, and Precision@k about practical utility.

## Training Strategy

### The Training Loop (Graph Version)

```python
def train_epoch(model, data):
    model.train()
    optimizer.zero_grad()
    
    # Difference from traditional ML:
    # Model takes BOTH features AND graph structure
    output = model(data.x, data.edge_index)  # edges matter!
    
    # Same loss computation as traditional ML
    loss = F.cross_entropy(output[data.train_mask], data.y[data.train_mask])
    
    # Backward pass propagates through:
    # 1. Neural weights
    # 2. Message passing operations  
    # 3. Graph connectivity (implicitly)
    loss.backward()
    optimizer.step()
```

Key Insight: The training looks familiar, but *model(data.x, data.edge_index)* is fundamentally different. The *edge_index* parameter enables information flow between related papers, which is the essence of Graph ML.

# Results

![Training Results](images/training_results_comprehensive.png?raw=true)


## t-SNE Visualization

![t-SNE Embeddings](images/tsne_node_embeddings.png?raw=true)

*t-SNE visualization shows the node encoder learned meaningful clusters. Papers from the same research area embed close together, demonstrating successful graph learning.*

## What the Model Learned

1. Content matters: Word vectors help identify paper topics
2. Citations matter: Papers citing similar papers are likely similar
3. Community structure: Research areas form natural clusters
4. Bridge papers: Some papers connect different research communities

# Practical Applications

## For Researchers

1. Literature discovery: Find papers you should cite but missed
2. Research mapping: Identify connections between subfields
3. Anomaly detection: Find papers that don't cite expected references

## For Publishers & Conferences

1. Reviewer matching: Find experts based on citation patterns
2. Paper recommendation: "Papers like this" functionality
3. Trend analysis: Detect emerging research areas

## For Libraries & Databases

1. Automated tagging: Classify new papers without manual labeling
2. Knowledge graph enrichment: Discover missing citations
3. Personalized search: Rank papers by relevance to user's interests

# Conclusion

Graph Neural Networks offer a paradigm shift for analyzing relational data. In citation networks:

1. Structure matters: Citations provide information beyond paper content
2. GraphSAGE works: Inductive learning handles real-world dynamics
3. Multi-task needs care: Separate encoders prevent interference
4. Evaluation is nuanced: Ranking metrics matter for link prediction

The Cora dataset, while small, demonstrates fundamental principles that scale to massive citation networks.

# References

- [GraphSAGE Paper](https://arxiv.org/abs/1706.02216): Original paper by Hamilton et al.
- [PyG Documentation](https://pyg.org/): Tutorials and examples
- [Cora Dataset](https://www.researchgate.net/publication/2947682_Automating_the_Construction_of_Internet_Portals): McCallum et al. "Automating the Construction of Internet Portals" (2000)