import math
import matplotlib.pyplot as plt
import networkx as nx

# helper function to simulate infinity, used for unreachable nodes
INF = float('inf')

def read_graph(filename):
    edges = [] #to store edges in the format (source, destination, distance)
    nodes = set() #to track distinct nodes
    
    with open(filename, 'r', newline='', encoding='utf-8') as f:
        header = f.readline() #skip header line in the file
        for line in f:
            source, destination, dist_str, _ = line.strip().split(',')
            dist = float(dist_str) #convert the distance to float
            edges.append((source.strip(), destination.strip(), dist)) #add the edge to the list
            nodes.add(source.strip()) #add source node to the set
            nodes.add(destination.strip()) #add destination node to the set
    
    return edges, nodes

#implement the Floyd-Warshall algorithm to find the shortest paths between all pairs of nodes
def floyd_warshall(nodes, edges):
    node_list = sorted(nodes) #sort nodes alphabetically for consistent indexing
    n = len(node_list) #number of nodes
    index = {node: i for i, node in enumerate(node_list)} #mapping node names to indices
    dist = [[INF]*n for _ in range(n)] #initialize distance matrix with "infinity" or no path
    
    for i in range(n):
        dist[i][i] = 0.0 #distance from a node to itself is zero
    
    next_hop = [[None]*n for _ in range(n)] #initialize the next hop matrix with None (no next node)
    
    #initialize the distance matrix and next hop matrix using the edges
    for s, d, w in edges:
        i, j = index[s], index[d]
        dist[i][j] = w #set the distance between source and destination
        next_hop[i][j] = j #set the next hop to be the destination
    
    #Floyd-Warshall - update the distance matrix with the shortest paths
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j] #update shortest distance
                    next_hop[i][j] = next_hop[i][k] #update the next hop
    
    return dist, next_hop, node_list

#reconstructs the shortest path from node i to node j using the next_hop matrix
def reconstruct_path(i, j, next_hop, node_list):
    if next_hop[i][j] is None:
        return [] #return an empty path if there is no path between i and j
    
    path = [node_list[i]] #initialize the path with the source node
    while i != j:
        i = next_hop[i][j] #move to the next node in the shortest path
        path.append(node_list[i]) #add the node to the path
    
    return path

#finds the diameter of the graph (the longest shortest path) and the path of it
def find_diameter(dist, next_hop, node_list):
    diameter = -1.0 #initialize diameter to a very small value
    diameter_path = [] #list to store the nodes in the diameter path
    n = len(node_list)
    
    #iterate over all pairs of nodes to find the longest shortest path
    for i in range(n):
        for j in range(n):
            if i != j and dist[i][j] != INF: #avoid paths to the same node or unreachable paths
                if dist[i][j] > diameter:
                    diameter = dist[i][j] #update diameter if a longer path is found
                    diameter_path = reconstruct_path(i, j, next_hop, node_list) #get the path for the diameter
    
    return diameter, diameter_path

#applies a log transformation to the distance matrix andapperantlythis helps with vizualization
def log_transform(dist):
    n = len(dist)
    epsilon = 1e-9 #small value to avoid log(0) (because it's undefined)
    
    for i in range(n):
        for j in range(n):
            val = dist[i][j]
            if val <= 0:
                val = epsilon #ensure that we don't take log of zero
            dist[i][j] = math.log(val + epsilon) #apply the logarithm transformation
    
    return dist

#writes the diameter and the path of the longest shortest path to a file.
def write_results_to_file(diameter, diameter_path, filename="diameter_and_path.txt"):
    with open(filename, "w") as file:
        file.write(f"Diameter: {diameter} light years\n") #write diameter to the file
        file.write("Path of the longest shortest path (Diameter Path):\n")
        file.write(" -> ".join(diameter_path)) #write the path as a string with ' -> ' between nodes

#plots the graph 
def plot_graph_with_path_and_labels(nodes, edges, diameter_path, node_list):
    #create a graph object
    G = nx.Graph()
    G.add_weighted_edges_from(edges) #add edges with weights to the graph
    
    #create a mapping of node labels for display purposes
    pos = nx.kamada_kawai_layout(G) #Kamada-Kawai layout for better spacing (suggested)
    
    #prepare edge labels (the lightyears)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    
    #create a list of edges that form the diameter path
    diameter_edges = [(diameter_path[i], diameter_path[i + 1]) for i in range(len(diameter_path) - 1)]
    
    plt.figure(figsize=(12, 12))  
    
    #draw the graph with node labels
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=12, font_weight='bold')
    
    #draw edge labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, font_color="black")
    
    #highlight the diameter path in red
    nx.draw_networkx_edges(G, pos, edgelist=diameter_edges, edge_color='red', width=3)
    
    plt.title("Cosmic Web with Diameter Path Highlighted in Red")
    plt.show() #show it

if __name__ == "__main__":
    filename = "../actual_distances_space_graph.csv" 
    edges, nodes = read_graph(filename) 
    dist_matrix, next_hop, node_list = floyd_warshall(nodes, edges) 
    diameter, diameter_path = find_diameter(dist_matrix, next_hop, node_list)  
    
    #output the results to a text file
    write_results_to_file(diameter, diameter_path)
    
    #print the results to the console
    print(f"Diameter (Longest Shortest Path Distance): {diameter} light years")
    print(f"Diameter Path: {' -> '.join(diameter_path)}")
    
    #transform the distance matrix for better visualization 
    dist_matrix = log_transform(dist_matrix)
    
    #plot the graph with the diameter path highlighted
    plot_graph_with_path_and_labels(nodes, edges, diameter_path, node_list)


"""
This programm reads a CSV file to construct a graph with nodes and weighted edges 
representing distances between them. It implements the Floyd-Warshall algorithm to compute 
the shortest paths between all pairs of nodes. The code also finds the graph's diameter 
(the longest shortest path) and reconstructs the path for that diameter. It applies a logarithmic 
transformation to the distance matrix for better visualization, writes the diameter and the path 
to a file, and generates a plot of the graph with the diameter path highlighted in red.

Time complexity - O(N^3) where N is the number of nodes
Memory complexity - O(N^2 + E) where N is the number of nodes and E is number of edges
"""