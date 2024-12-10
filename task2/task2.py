import networkx as nx
import csv
import matplotlib.pyplot as plt

#directed graph from the .csv file
edges = []

# Read the CSV file and create the graph
with open('../actual_distances_space_graph.csv', newline='', encoding='utf-8') as csvfile:
    csvreader = csv.DictReader(csvfile)
    for row in csvreader:
        source = row['source']
        destination = row['destination']
        hyperflow = float(row['hyperflowSpiceMegaTons'])
        edges.append((source, destination, hyperflow))

#directed graph
G = nx.DiGraph()
G.add_weighted_edges_from(edges, weight='capacity')

#maximum flow from Earth to Betelgeuse using Edmonds-Karp
flow_value, flow_dict = nx.maximum_flow(G, 'Earth', 'Betelgeuse', flow_func=nx.algorithms.flow.edmonds_karp)

#print the flow value
print(f"Maximum flow from Earth to Betelgeuse: {flow_value} Spice MegaTons")

#print the flow distribution
print("Flow distribution:", flow_dict)

#create a list of edge labels with flow values
edge_labels = {}
for u, v, data in G.edges(data=True):
    flow = flow_dict[u].get(v, 0)
    if flow > 0:  #only label edges with actual values
        edge_labels[(u, v)] = str(int(flow))  #convert to integer so it looks better

plt.figure(figsize=(14, 14))  #size of the plot

pos = nx.spring_layout(G, k=1.5, seed=42) 

#draw the graph with node size, font size, and color
nx.draw(G, pos, with_labels=True, node_size=2000, font_size=12, node_color='lightblue', font_weight='bold', edge_color='gray')

#edge labels
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

plt.title('Flow Network from Earth to Betelgeuse', fontsize=16)

plt.show() #show it


"""
The programm reads a CSV file to create a directed graph using the networkx library, where edges represent 
the flow between nodes, and weights denote the capacity of each edge. It then calculates the maximum flow 
from Earth to Betelgeuse using the Edmonds-Karp algorithm, which is an implementation of the Ford-Fulkerson 
method for computing the maximum flow in a flow network.

Time complexity - O(V * E^2) where V is the number of nodes and E is the number of edges
Memory complexity - O(V + E) where N is the number of nodes
"""
