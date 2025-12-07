# Node Classification and Link Prediction in Citation Networks using Graph Neural Networks

*December 2025 · 15 min read · [Open in Colab](https://colab.research.google.com/github/BogGoro/predicting-paper-topics-and-connections-tutorial/blob/main/GraphML_Tutorial.ipynb)*

# Introduction

Graph Machine Learning (Graph ML) is revolutionizing how we analyze interconnected data. From social networks to molecular structures and citation networks, graphs are everywhere. In this comprehensive tutorial, we implement and compare two state-of-the-art Graph Neural Network architectures GraphSAGE and GATv2 applied to the classic problem of academic citation network analysis.

We'll build a system that can:

1. **Classify research papers** into topics (Node Classification)
2. **Predict potential citations** between papers (Link Prediction)

All using the powerful **PyTorch Geometric** library and real-world data from the Cora dataset, with extensive comparative analysis across 100 independent runs.

# Why This Matters

Citation networks present unique challenges:

* **Relational data**: Papers influence each other through citations
* **Homophily**: Similar papers tend to cite each other
* **Semi-supervised setting**: We have labels for only some papers

Traditional ML struggles with graph data because it ignores relationships between data points. GNNs solve this by learning from both node features **and** graph structure through specialized mechanisms like neighborhood aggregation and attention.

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

# Architectures Compared: GraphSAGE vs GATv2

## GraphSAGE (SAmple and aggreGatE)

We implement GraphSAGE for several reasons:

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

## GATv2 (Graph Attention Network v2)

GATv2 represents a significant advancement over the original GAT by implementing dynamic attention that overcomes the static attention limitations of GAT. We implement a comprehensive GATv2 system with separate encoders for node classification and link prediction.

### Why GATv2 Over Original GAT?

The original GAT computes attention scores as:
$\alpha_{ij} = \text{softmax}_j(\text{LeakyReLU}(\mathbf{a}^\top [\mathbf{W}\mathbf{h}_i \Vert \mathbf{W}\mathbf{h}_j]))$

This formulation has a critical limitation: after applying the softmax, the ranking of attention scores for a given node $i$ is **fixed** regardless of the query node. This makes it **static attention**.

GATv2 fixes this by computing:
$\alpha_{ij} = \text{softmax}_j(\mathbf{a}^\top \text{LeakyReLU}(\mathbf{W}[\mathbf{h}_i \Vert \mathbf{h}_j]))$

Now the attention mechanism is dynamic - the ranking can change depending on the query node, making it strictly more expressive.

### Our GATv2 Implementation

```python
class GATv2Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.2):
        super().__init__()
        from torch_geometric.nn import GATv2Conv
        
        # First layer: Multi-head attention for rich feature extraction
        self.conv1 = GATv2Conv(
            in_channels, hidden_channels, 
            heads=heads, dropout=dropout, concat=True
        )
        
        # Second layer: Attention refinement
        self.conv2 = GATv2Conv(
            hidden_channels * heads, out_channels,
            heads=1, dropout=dropout, concat=False
        )
        
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None  # Store for analysis
```

### Mathematical Details of GATv2

For each attention head $h$ at layer $l$:

1. **Linear Transformation**
$\mathbf{h}_i' = \mathbf{W}^l \mathbf{h}_i$

2. **Dynamic Attention Computation**
$e_{ij}^h = \text{LeakyReLU}\left( \mathbf{a}^{\top} \left[ \mathbf{h}_i' \Vert \mathbf{h}_j' \right] \right)$

3. **Normalized Attention Coefficients**
$\alpha_{ij}^h = \frac{\exp(e_{ij}^h)}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik}^h)}$

4. **Aggregation**
$\mathbf{h}i^{l+1} = \sigma\left( \frac{1}{H} \sum{h=1}^{H} \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^h \cdot \mathbf{h}_j' \right)$

where:
- $\mathbf{a}$: learnable attention vector
- $H$: number of attention heads
- $\mathcal{N}(i)$: neighbors of node $i$

### Multi-Head Attention in GATv2

We use 8 attention heads in our implementation, which allows the model to:

1. **Capture different types of relationships** (e.g., citation vs content similarity)
2. **Stabilize training** through ensemble-like behavior
3. **Increase model capacity** without significantly increasing parameters

Each head learns different attention patterns, and their outputs are concatenated (early layers) or averaged (final layer).

### GATv2 for Link Prediction

For link prediction, we implement a specialized GATv2 encoder:

```python
class GATv2LinkEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.2):
        super().__init__()
        # Three-layer architecture for link prediction
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, concat=True)
        self.conv2 = GATv2Conv(hidden_channels*heads, hidden_channels, heads=heads, concat=True)
        self.conv3 = GATv2Conv(hidden_channels*heads, out_channels, heads=1, concat=False)
```

This deeper architecture allows the model to learn:

1. Layer 1: Initial feature transformation with attention
2. Layer 2: Neighborhood aggregation focusing on citation-relevant patterns
3. Layer 3: Final embeddings optimized for link existence prediction

# Multi-Task Architecture

## The Dual-Encoder Solution

We implement separate encoders for node classification and link prediction to prevent task interference:

```python
class GraphSAGEModels(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        # Separate encoders for separate tasks
        self.node_encoder = NodeEncoder(...)    # Classification-optimized
        self.link_encoder = LinkEncoder(...)    # Link-prediction-optimized
        
    def forward_node(self, x, edge_index):
        z_node = self.node_encoder(x, edge_index)
        return self.node_classifier(z_node, edge_index)
    
    def forward_link(self, x, edge_index):
        z_link = self.link_encoder(x, edge_index)
        return self.link_predictor(z_link[src], z_link[dst])
```

Why Two Encoders Work Better:

- **Node classification encoder**: Learns discriminative features that separate research areas
- **Link prediction encoder**: Learns similarity-preserving features for citation prediction
- **No gradient conflict**: Each encoder optimizes for its specific objective
- **Specialized architectures**: Different depths and aggregation strategies per task

# Link Prediction Challenges

## The Negative Sampling Problem

Citation networks present extreme class imbalance:

```python
from torch_geometric.utils import negative_sampling

# For 2,708 papers: 7.3 million possible citations
# Only 10,556 are observed (0.14% positive rate)

neg_edges = negative_sampling(
    edge_index=data.edge_index,
    num_nodes=data.num_nodes,           # 2708
    num_neg_samples=pos_edges.size(1)   # Balance with positives
)
```

### Why Link Prediction is Particularly Challenging

- Extreme sparsity: 10,556 observed edges vs 7.3M possible edges
- Missing not at random: Unobserved edges might be "won't cite" or "haven't cited yet"
- Transductive evaluation: Must test on the same graph but unseen edges

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

## 100 Independent Runs

To ensure statistical significance and robustness:

```python
def run_multiple_experiments(data, link_data, num_runs=100):
    results = {"GraphSAGE": [], "GATv2": []}
    for run in range(num_runs):
        # Different random seed each run
        torch.manual_seed(42 + run)
        np.random.seed(42 + run)
        
        # Train and evaluate both models
        sage_results = train_evaluate_graphsage(...)
        gatv2_results = train_evaluate_gatv2(...)
        
        # Store for statistical analysis
        results["GraphSAGE"].append(sage_results)
        results["GATv2"].append(gatv2_results)
```

# Results

## Comprehensive Performance Comparison

After 100 independent runs:

|Model|Node Accuracy|Link AUC|Link AP|Precision@100|Training Time (s)|
|:----|:------------|:-------|:------|:------------|:----------------|
|GraphSAGE|0.7561 ± 0.0171 [0.7050-0.7940]|0.7086 ± 0.0191 [0.6673-0.7608]|0.7229 ± 0.0221 [0.6742-0.7751]|0.8619 ± 0.0367 [0.7800-0.9500]|1.1 ± 0.1 [1.1-1.4]|
|GATv2|0.7368 ± 0.0228 [0.6360-0.7790]|0.8123 ± 0.0165 [0.7492-0.8406]|0.8088 ± 0.0176 [0.7416-0.8410]|0.9069 ± 0.0269 [0.8200-0.9500]|7.2 ± 0.1 [7.1-8.1]|

![Training Results](images/multiple_runs_comparison.png?raw=true)

## Key Findings

1. Task-Specific Superiority:

- GraphSAGE outperforms GATv2 in Node Classification (75.6% vs 73.7%, +1.9% absolute improvement)
- GATv2 significantly outperforms GraphSAGE in Link Prediction (AUC: 81.2% vs 70.9%, +10.3% absolute improvement)

2. Computational Efficiency:

- GraphSAGE trains 6.5× faster than GATv2 (1.1s vs 7.2s per run)
- GATv2's attention mechanism provides substantial link prediction benefits at significant computational cost

3. Model Stability:

- GraphSAGE shows more consistent node classification performance (±1.7% std vs ±2.3% for GATv2)
- GATv2 demonstrates more stable link prediction results (±1.7% std vs ±1.9% for GraphSAGE)

## Research Insights

1. Architecture-task alignment: Different GNN architectures have inherent biases toward specific graph tasks
2. Attention for relationships: GATv2's attention mechanism is particularly effective for learning edge existence patterns
3. Efficiency-performance trade-off: Clear demonstration of the compute-accuracy trade-off in graph ML

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

This comprehensive study demonstrates that different Graph Neural Network architectures excel at different tasks in citation network analysis. Through rigorous experimentation across 100 independent runs, we've uncovered nuanced performance patterns that challenge the notion of a universal "best" GNN architecture.

# References

- [GraphSAGE Paper](https://arxiv.org/abs/1706.02216): Original paper by Hamilton et al.
- [GATv2 Paper](https://arxiv.org/abs/2105.14491): Original paper by Brody et al.
- [PyG Documentation](https://pyg.org/): Tutorials and examples
- [Cora Dataset](https://www.researchgate.net/publication/2947682_Automating_the_Construction_of_Internet_Portals): McCallum et al. "Automating the Construction of Internet Portals" (2000)