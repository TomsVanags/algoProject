import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load the data
file_path = "Finalpro/actual_distances_space_graph.csv"
df = pd.read_csv(file_path)

# Create an edge list from the dataframe
edges = []
for _, row in df.iterrows():
    edges.append((row['source'], row['destination'], row['hyperflowSpiceMegaTons']))

# Kruskal's Algorithm
def find(parent, node):
    if parent[node] != node:
        parent[node] = find(parent, parent[node])  # Path compression
    return parent[node]

def union(parent, rank, node1, node2):
    root1 = find(parent, node1)
    root2 = find(parent, node2)
    
    if root1 != root2:
        if rank[root1] > rank[root2]:
            parent[root2] = root1
        elif rank[root1] < rank[root2]:
            parent[root1] = root2
        else:
            parent[root2] = root1
            rank[root1] += 1

def kruskal(nodes, edges):
    mst = []
    total_weight = 0

    # Initialize parent and rank dictionaries
    parent = {node: node for node in nodes}
    rank = {node: 0 for node in nodes}

    # Sort edges by weight
    edges.sort(key=lambda x: x[2])

    for edge in edges:
        u, v, weight = edge
        if find(parent, u) != find(parent, v):
            union(parent, rank, u, v)
            mst.append((u, v, weight))
            total_weight += weight

    return mst, total_weight

# Extract unique nodes
nodes = set(df['source']).union(set(df['destination']))

# Run Kruskal's Algorithm
mst_edges, total_weight = kruskal(nodes, edges)

# Output the results
print("Total Weight of the Minimum Spanning Tree:", total_weight)
print("Edges in the Minimum Spanning Tree:")
for edge in mst_edges:
    print(f"{edge[0]} -> {edge[1]} with weight {edge[2]}")

# Save the MST to a file
output_file = "QMST_edges_manual.csv"
mst_df = pd.DataFrame(mst_edges, columns=['source', 'destination', 'weight'])
mst_df.to_csv(output_file, index=False)

# Convert MST to a NetworkX graph for visualization
mst_graph = nx.Graph()
for u, v, weight in mst_edges:
    mst_graph.add_edge(u, v, weight=weight)

# Visualize the MST
plt.figure(figsize=(12, 10))
pos = nx.spring_layout(mst_graph, k=0.25, iterations=100)  # Spring layout for positioning

# Draw the graph nodes and edges
nx.draw(
    mst_graph,
    pos,
    with_labels=True,
    node_color="lightblue",
    edge_color="gray",
    node_size=3000,
    font_size=12,
)

# Adjust edge labels for better visibility
edge_labels = {(u, v): f"{d['weight']}" for u, v, d in mst_graph.edges(data=True)}
nx.draw_networkx_edge_labels(
    mst_graph,
    pos,
    edge_labels=edge_labels,
    font_color="red",
    font_size=10,  # Adjust font size for edge labels
    label_pos=0.5,  # Position labels at the center of edges
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7)  # Add background to labels for better contrast
)

# Add padding to ensure labels do not overlap or fall outside the visible area
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

plt.title("Quantum Minimum Spanning Tree (QMST)")
plt.show()

# Iterating through the CSV file and adding edges is O(n)
# Kruskals algorithm takes O(nlogn)
# use graph theory to optimize and analyze a communication network
# Achieve efficient communication across the galactic network with minimal resource usage.

"""Output: Total Weight of the Minimum Spanning Tree: 1876
Edges in the Minimum Spanning Tree:
Mars -> Sirius with weight 62
Arcturus -> Earth with weight 78
Ganymede -> Europa with weight 102
Earth -> Mars with weight 131
Mars -> Betelgeuse with weight 138
Vega -> Earth with weight 154
Earth -> AlphaCentauri with weight 163
ProximaCentauri -> AlphaCentauri with weight 172
Mars -> Europa with weight 215
Polaris -> Vega with weight 324
Andromeda -> Arcturus with weight 337"""
