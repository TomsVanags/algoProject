import networkx as nx 
import csv 
import matplotlib.pyplot as plt

# function for Welsh-Powell algorithm
def welsh_powell_coloring(graph):
    #sort nodes by degree by DESC
    sorted_nodes = sorted(graph.nodes(), key=lambda node: len(list(graph.neighbors(node))), reverse=True)
    
    #store colors of each node
    node_colors = {}
    
    #color the graph
    for node in sorted_nodes:
        #color of neighbour ndes
        neighbor_colors = {node_colors.get(neighbor) for neighbor in graph.neighbors(node)}
        
        #find smallest color that is not used by neighbors
        color = 0
        while color in neighbor_colors:
            color += 1
        
        # assign the color
        node_colors[node] = color
    
    return node_colors

edges = []

#read the file and make the graph
with open('../actual_distances_space_graph.csv', newline='', encoding='utf-8') as csvfile:
    csvreader = csv.DictReader(csvfile)
    for row in csvreader:
        source = row['source']
        destination = row['destination']
        edges.append((source, destination))

G = nx.Graph()
G.add_edges_from(edges)

#use Welsh-Powellh
coloring = welsh_powell_coloring(G)

#print count of colors used
num_colors = max(coloring.values()) + 1
print(f"Number of colors used: {num_colors}")

#print mapping
print("Node-to-color mapping:", coloring)

# create a list of colors
node_color = [coloring[node] for node in G.nodes()]

#I tested it, and we only need 4 colors, so I only added 4
color_palette = ['red', 'blue', 'green', 'yellow'] 

#ensure the palette size is enough
color_palette = color_palette[:num_colors]  # we can also trim it. And this is not REALLY needed, but his in case there are not enough or too many colors in color_pallete

#map the colors
node_color = [color_palette[color] for color in node_color]

#draw the graph with the assigned colors
nx.draw(G, with_labels=True, node_color=node_color, node_size=500, font_size=10)

plt.show() #show it

"""
This applies the Welsh-Powell algorithm to color the nodes of a graph. 
It first sorts the nodes by their degree in descending order, then assigns the smallest 
available color to each node, ensuring that no two adjacent nodes share the same color. 
The graph is constructed from a CSV file, and the coloring results are visualized using a 
color palette. The code also outputs the number of colors used and the node-to-color mapping.

Time complexity - O(N Log N + E) where N is the number of nodes and E is the number of edges
Memory complexity - O(N + E)
"""
