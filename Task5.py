import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def read_graph(file_path):
    # Read from a CSV file and create an undirected graph
    df = pd.read_csv(file_path) # easy reading with pandas library
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row['source'], row['destination'], weight=row['distanceLY'])
    return G

def find_cycles(graph):
    # Find all cycles in the graph using DFS(Depth-First Search)
    # If DFS encounters a node that has already been visited and is not the parent of the current node, a cycle exists.
    cycles = []
    for cycle in nx.cycle_basis(graph): # in NetworkX's built in method for DFS
        cycles.append(cycle)
    return cycles

def calculate_cycle_weight(graph, cycle):
    # Calculate the total weight of a cycle
    weight = 0 # weight is the sum of the weights of all edges that make up the cycle
    for i in range(len(cycle)):
        u = cycle[i] # each node in cycle
        v = cycle[(i + 1) % len(cycle)] # ensures that last node connects to first
        weight += graph[u][v]['weight'] # access weight between u and v and add to total weight
    return weight

def find_longest_loop(graph, cycles):
    # Find the longest cycle by weight
    max_weight = 0
    longest_cycle = None
    for cycle in cycles:
        weight = calculate_cycle_weight(graph, cycle) #calculate each cycles weight with previous function
        if weight > max_weight:
            max_weight = weight
            longest_cycle = cycle
    return longest_cycle, max_weight

def visualize_cycle(graph, cycle, title):

    subgraph = graph.subgraph(cycle)
    pos = nx.spring_layout(subgraph)
    plt.figure(figsize=(8, 6))
    nx.draw(subgraph, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=12)
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in subgraph.edges(data=True)}
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_color='red')
    plt.title(title)
    plt.show()

# Main Program
file_path = "Finalpro/actual_distances_space_graph.csv"

# Step 1: Read the graph
graph = read_graph(file_path)

# Step 2: Find all cycles
cycles = find_cycles(graph)

# Step 3: Find the longest cycle
longest_cycle, max_weight = find_longest_loop(graph, cycles)

# Step 4: Output the results
print("Longest Cycle:", longest_cycle)
print("Sum of Weights in Longest Loop:", max_weight)

# Step 5: Visualize the longest cycle
if longest_cycle:
    visualize_cycle(graph, longest_cycle, "Longest Cycle in the Space-Time Fabric")
else:
    print("No cycles found in the graph.")


    # Weight corresponds to distanceLY (the distance between two points in light-years).
    # NetworkX computes the graph structure. NetworkX internally uses Kruskal's or Borůvka’s algorithms
    # Matplotlib is used to draw the nodes and edges using coordinates.
    # MST connects all nodes with the minimum total weight while avoiding cycles.
    # Kruskals algorithm uses O(nlogn) complexity where n is number of edges
    # could use quantum-inspired algorithms or optimizations for QMST.
