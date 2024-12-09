import csv        # Lets us read data from CSV files
import math       # Gives us infinite values, math operations, and other helpful math tools
import networkx as nx  # A library for working with graphs (nodes, edges, algorithms)
import matplotlib.pyplot as plt  # For drawing and visualizing the graph
import numpy as np   # Useful for numerical operations and arrays, if needed
from sklearn.manifold import MDS  # Helps place nodes in a space so distances represent our data

def read_graph(filename):
    # Reads the CSV file and collects edges and nodes from it.
    # Each row in the CSV gives us source, destination, and the distance between them.
    edges = []      # Will store tuples (source, destination, distance)
    nodes = set()   # Will keep track of all distinct nodes we encounter
    with open(filename, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)     # Create a CSV reader object
        header = next(reader)      # Skip the first line (header)
        for row in reader:
            # row should contain something like: source, destination, distanceLY, hyperflow
            source, destination, dist_str, _ = row
            dist = float(dist_str)   # Convert the distance from string to float
            edges.append((source.strip(), destination.strip(), dist))  # Store edge info
            nodes.add(source.strip())    # Add the source to the set of nodes
            nodes.add(destination.strip())  # Add the destination to the set of nodes
    return edges, nodes  # Return the list of edges and the set of nodes

def floyd_warshall(nodes, edges):
    # Applies the Floyd-Warshall algorithm to find shortest paths between every pair of nodes.
    # It returns:
    # dist: a matrix of shortest distances between all pairs of nodes
    # next_hop: a matrix to help us reconstruct the shortest path
    # node_list: a sorted list of nodes so indices match the matrix rows/columns

    node_list = sorted(nodes)               # Sort nodes to have a consistent order
    n = len(node_list)                      # Count how many nodes we have
    index = {node: i for i, node in enumerate(node_list)} # Map each node to its index in the matrix

    # Initialize a distance matrix with infinity for all pairs except zero on the diagonal
    dist = [[math.inf]*n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0.0   # Distance from a node to itself is zero

    # next_hop will help us reconstruct paths after we compute them
    next_hop = [[None]*n for _ in range(n)]

    # Fill in the direct distances we know from the edges
    for s, d, w in edges:
        i, j = index[s], index[d]
        dist[i][j] = w
        next_hop[i][j] = j

    # Floyd-Warshall: try to improve distances by going through intermediate nodes
    for k in range(n):
        for i in range(n):
            for j in range(n):
                # If going from i to k and then k to j is better than the current known distance i->j, update it
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next_hop[i][j] = next_hop[i][k]

    return dist, next_hop, node_list

def reconstruct_path(i, j, next_hop, node_list):
    # Reconstructs the actual path of nodes from i to j using next_hop.
    # If there's no path, returns an empty list.
    if next_hop[i][j] is None:
        return []
    path = [node_list[i]]     # Start from the beginning node
    while i != j:
        i = next_hop[i][j]
        path.append(node_list[i])
    return path

def find_diameter(dist, next_hop, node_list):
    # Finds the diameter of the graph: the longest shortest path between any two nodes.
    # diameter: the length (in whatever metric we're using, here light years) of the longest shortest path
    # diameter_path: the actual sequence of nodes for that longest shortest path
    n = len(node_list)
    diameter = -1.0
    diameter_path = []
    for i in range(n):
        for j in range(n):
            # Only consider pairs with a finite distance and that are not the same node
            if i != j and dist[i][j] != math.inf:
                # If we found a longer shortest path, record it
                if dist[i][j] > diameter:
                    diameter = dist[i][j]
                    diameter_path = reconstruct_path(i, j, next_hop, node_list)
    return diameter, diameter_path

def make_symmetric_and_finite(dist):
    # The MDS method requires a symmetric distance matrix.
    # This function makes the distance matrix symmetric by:
    # - Replacing infinite values with a large finite number
    # - Ensuring dist[i][j] == dist[j][i] by averaging
    n = len(dist)
    large_value = 3e7  # A large finite number to replace infinities
    for i in range(n):
        for j in range(n):
            if math.isinf(dist[i][j]) and math.isinf(dist[j][i]):
                dist[i][j] = large_value
                dist[j][i] = large_value
            elif math.isinf(dist[i][j]) and not math.isinf(dist[j][i]):
                dist[i][j] = dist[j][i]
            elif not math.isinf(dist[i][j]) and math.isinf(dist[j][i]):
                dist[j][i] = dist[i][j]

    # Make sure dist[i][j] == dist[j][i] by taking the average
    for i in range(n):
        for j in range(i+1, n):
            val = (dist[i][j] + dist[j][i]) / 2.0
            dist[i][j] = val
            dist[j][i] = val
    return dist

def log_transform(dist):
    # Applies a logarithmic transformation to distances.
    # This reduces the impact of very large distances, making the visualization better.
    n = len(dist)
    epsilon = 1e-9  # A tiny number to avoid log(0)
    for i in range(n):
        for j in range(n):
            val = dist[i][j]
            if val <= 0:
                val = epsilon
            dist[i][j] = math.log(val + epsilon)
    return dist

def plot_graph_with_path_and_labels(G, coords, node_list, diameter_path, edges, title):
    # Draws the graph, labels nodes, and shows distances on edges.
    # Also highlights the diameter path in red so it's easy to see.

    # Create a position dictionary: node -> (x, y)
    pos = {node_list[i]: coords[i] for i in range(len(node_list))}

    # Identify the edges on the diameter path
    diameter_edges = []
    for i in range(len(diameter_path) - 1):
        diameter_edges.append((diameter_path[i], diameter_path[i+1]))

    plt.figure(figsize=(10,10)) # Make the plot a nice size

    # Draw all the nodes in light blue
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
    # Draw labels for each node so we know which is which
    nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')

    # Draw all edges in gray, so we can easily spot the red path edges
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=10, edge_color='gray')

    # Create edge labels based on the original distances from edges
    edge_labels = {(u,v): f"{w} ly" for u,v,w in edges}
    # Put the edge labels on the plot
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

    # Now overlay the diameter path edges in red, to make it stand out
    nx.draw_networkx_edges(G, pos, edgelist=diameter_edges, arrowstyle='->', arrowsize=15, edge_color='red', width=2)

    # Add a title to the plot
    plt.title(title)
    # Hide the axis since this is not a map, just a visualization
    plt.axis('off')
    # Adjust layout so nothing is cut off
    plt.tight_layout()
    # Show the final plot
    plt.show()

# This is where the main program runs
if __name__ == "__main__":
    filename = "actual_distances_space_graph.csv"  # The CSV file containing our graph
    edges, nodes = read_graph(filename)            # Read the graph from the CSV
    dist_matrix, next_hop, node_list = floyd_warshall(nodes, edges) # Compute shortest paths
    diameter, diameter_path = find_diameter(dist_matrix, next_hop, node_list) # Find the diameter and its path

    # Print the diameter and the path
    # This is important for the task that requires identifying the longest shortest path
    print("Diameter (Longest Shortest Path Distance):", diameter, "light years")
    print("Diameter Path:", " -> ".join(diameter_path))

    # Make the distance matrix symmetric and finite, required for MDS
    dist_matrix = make_symmetric_and_finite(dist_matrix)
    # Apply a log transform to help space out nodes better
    dist_matrix = log_transform(dist_matrix)

    # Create a NetworkX graph from the edges
    G = nx.DiGraph()
    for s, d, w in edges:
        G.add_edge(s, d, weight=w)

    # Use MDS to get coordinates for each node based on their distances
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coords_mds = mds.fit_transform(dist_matrix)

    # Finally, plot the graph with the computed layout, showing edge labels and highlighting the diameter path
    plot_graph_with_path_and_labels(G, coords_mds, node_list, diameter_path, edges, "MDS with Log-Transformed Distances (Diameter Path in Red)")


# Additional Notes:
# How to run:
#   Just run this Python script in a terminal or IDE where you have Python installed.
#   Make sure you have the CSV file "actual_distances_space_graph.csv" in the same directory.
#
# What pip installs do you need?
#   pip install matplotlib networkx scikit-learn
#   These three packages should cover everything used here.
#
# What does this code do?
#   - Reads a CSV of interstellar edges (like planets connected by routes).
#   - Computes shortest paths between all nodes.
#   - Finds the longest shortest path (the diameter) and prints it.
#   - Creates a visualization of the graph with MDS to space out nodes.
#   - Highlights the diameter path in red so it stands out.
#   - Shows distances on each edge (in light years).
#
# Complexity:
#   The Floyd-Warshall algorithm used here to compute shortest paths runs in O(n^3) time,
#   where n is the number of nodes. This is usually fine for smaller graphs.
#   MDS scaling complexity can vary depending on implementation and data size, 
#   but it can also be quite expensive for large n. 
