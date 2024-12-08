import networkx as nx
import csv
import matplotlib.pyplot as plt

# Create the directed graph from the .csv file data
edges = []

# Reading the CSV file and creating the graph
with open('actual_distances_space_graph.csv', newline='', encoding='utf-8') as csvfile:
    csvreader = csv.DictReader(csvfile)
    for row in csvreader:
        source = row['source']
        destination = row['destination']
        hyperflow = float(row['hyperflowSpiceMegaTons'])
        edges.append((source, destination, hyperflow))

# Initialize a directed graph
G = nx.DiGraph()
G.add_weighted_edges_from(edges, weight='capacity')

# Calculate the maximum flow from Earth to Betelgeuse using Edmonds-Karp
flow_value, flow_dict = nx.maximum_flow(G, 'Earth', 'Betelgeuse', flow_func=nx.algorithms.flow.edmonds_karp)

# Print the flow value
print(f"Maximum flow from Earth to Betelgeuse: {flow_value} Spice MegaTons")

# Print the flow distribution
print("Flow distribution:", flow_dict)

# Visualization: Draw the graph with flow values on edges
# Create a list of edge labels with flow values (display only flow value or capacity)
edge_labels = {}
for u, v, data in G.edges(data=True):
    flow = flow_dict[u].get(v, 0)
    if flow > 0:  # Only label edges with non-zero flow
        edge_labels[(u, v)] = str(int(flow))  # Convert to integer to avoid float display

# Increase figure size for better visibility
plt.figure(figsize=(14, 14))  # Adjust the size of the plot (larger for more space)

# Use a spring layout with adjusted k value for better spacing between nodes
pos = nx.spring_layout(G, k=1.5, seed=42)  # Increase k value to spread nodes apart more

# Draw the graph with customized node size, font size, and color
nx.draw(G, pos, with_labels=True, node_size=2000, font_size=12, node_color='lightblue', font_weight='bold', edge_color='gray')

# Draw the edge labels (only displaying integers for non-zero flow edges)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

# Display the graph with a title
plt.title('Flow Network from Earth to Betelgeuse', fontsize=16)
plt.show()

# Save the flow graph to a file (in GEXF format for better visualization)
nx.write_gexf(G, "flow_graph.gexf")
