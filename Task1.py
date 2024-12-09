import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load the data
file_path = "Finalpro/actual_distances_space_graph.csv"
df = pd.read_csv(file_path)

# Create a directed graph using NetworkX
G = nx.DiGraph()

# Add edges with weights (hyperflowSpiceMegaTons)
for _, row in df.iterrows():
    source = row['source']
    destination = row['destination']
    weight = row['hyperflowSpiceMegaTons']
    G.add_edge(source, destination, weight=weight)

# Compute the Minimum Spanning Tree
# Convert to undirected for MST calculation as NetworkX requires undirected graph for `minimum_spanning_tree`
mst = nx.minimum_spanning_tree(G.to_undirected(), weight='weight')

# Calculate the total weight of the MST
total_weight = sum(edge[2]['weight'] for edge in mst.edges(data=True))

# Output the results
print("Total Weight of the Minimum Spanning Tree:", total_weight)
print("Edges in the Minimum Spanning Tree:")
for edge in mst.edges(data=True):
    print(f"{edge[0]} -> {edge[1]} with weight {edge[2]['weight']}")

# Save the MST to a file
output_file = "QMST_edges.csv"
mst_edges = [(u, v, d['weight']) for u, v, d in mst.edges(data=True)]
mst_df = pd.DataFrame(mst_edges, columns=['source', 'destination', 'weight']) # get source, destination and edge.
mst_df.to_csv(output_file, index=False)


# Visualize the MST
plt.figure(figsize=(12, 10))
pos = nx.spring_layout(mst,k=0.25, iterations=100)  # determines the positioning of nodes in the visualization using a force-directed layout algorithm.

# Draw the graph nodes and edges
nx.draw(
    mst,
    pos,
    with_labels=True,
    node_color="lightblue",
    edge_color="gray",
    node_size=3000,
    font_size=12,
)

# Adjust edge labels for better visibility
edge_labels = {(u, v): f"{d['weight']}" for u, v, d in mst.edges(data=True)}
nx.draw_networkx_edge_labels(
    mst,
    pos,
    edge_labels=edge_labels,
    font_color="red",
    font_size=10,  # Adjust font size for edge labels
    label_pos=0.5,  # Position labels at the center of edges
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7)  # Add background to labels for better contrast
)

# Add padding to ensure labels do not overlap or fall outside the visible area
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

edge_labels = {(u, v): f"{d['weight']}" for u, v, d in mst.edges(data=True)}
nx.draw_networkx_edge_labels(mst, pos, edge_labels=edge_labels, font_color="red")
plt.title("Quantum Minimum Spanning Tree (QMST)")
plt.show()

# Total Weight of the Minimum Spanning Tree: 2179
# Iterating through the CSV file and adding edges is O(n)
# Kruskals algorithm takes O(nlogn)
# use graph theory to optimize and analyze a communication network
# Achieve efficient communication across the galactic network with minimal resource usage.

"""Output: Edges in the Minimum Spanning Tree:
Earth -> Arcturus with weight 78
Earth -> Mars with weight 131
Earth -> Vega with weight 154
Earth -> ProximaCentauri with weight 206
Earth -> Sirius with weight 322
Mars -> Betelgeuse with weight 138
Mars -> Europa with weight 215
Ganymede -> Europa with weight 102
AlphaCentauri -> ProximaCentauri with weight 172
Vega -> Polaris with weight 324
Arcturus -> Andromeda with weight 337"""
