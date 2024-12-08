import networkx as nx #for displaying the results
import csv #for reading th .csv file
import matplotlib.pyplot as plt

# Function to perform Welsh-Powell algorithm for graph coloring
def welsh_powell_coloring(graph):
    # Sort nodes by degree in descending order
    sorted_nodes = sorted(graph.nodes(), key=lambda node: len(list(graph.neighbors(node))), reverse=True)
    
    # Dictionary to store colors of each node
    node_colors = {}
    
    # Start coloring the graph
    for node in sorted_nodes:
        # Get the colors of the neighboring nodes
        neighbor_colors = {node_colors.get(neighbor) for neighbor in graph.neighbors(node)}
        
        # Find the smallest color that is not used by neighbors
        color = 0
        while color in neighbor_colors:
            color += 1
        
        # Assign the color to the node
        node_colors[node] = color
    
    return node_colors

# Create the graph from the .csv file data
edges = []

# Reading the CSV file and creating the graph
with open('actual_distances_space_graph.csv', newline='', encoding='utf-8') as csvfile:
    csvreader = csv.DictReader(csvfile)
    for row in csvreader:
        source = row['source']
        destination = row['destination']
        edges.append((source, destination))

# Initialize a graph
G = nx.Graph()
G.add_edges_from(edges)

# Apply Welsh-Powell algorithm to color the graph
coloring = welsh_powell_coloring(G)

# Print the number of colors used
num_colors = max(coloring.values()) + 1
print(f"Number of colors used: {num_colors}")

# Print the node-to-color mapping
print("Node-to-color mapping:", coloring)

# Visualization part
# Create a list of colors for the nodes based on the coloring dictionary
node_color = [coloring[node] for node in G.nodes()]

# Define a color palette with a sufficient number of colors
color_palette = ['red', 'blue', 'green', 'yellow']  # Added more colors

# Ensure the palette size is enough for the number of colors used
color_palette = color_palette[:num_colors]  # Trim the palette if necessary

# Map the colors based on the assigned color indices
node_color = [color_palette[color] for color in node_color]

# Draw the graph with customized colors
nx.draw(G, with_labels=True, node_color=node_color, node_size=500, font_size=10)

# Display the graph
plt.show()
