import pandas as pd  
import networkx as nx  
import matplotlib.pyplot as plt  

# load the CSV file
file_path = "../actual_distances_space_graph.csv"  # path
df = pd.read_csv(file_path)  #CSV data into a pandas DataFrame

#edge list from the DataFrame
edges = []
for _, row in df.iterrows():
    #row in the DataFrame represents an edge with a source, destination, and weight
    edges.append((row['source'], row['destination'], row['hyperflowSpiceMegaTons']))

# "find" function for the union-find algorithm (Kruskal's algorithm)
def find(parent, node):
    #finds the root of the node using path compression
    if parent[node] != node:
        parent[node] = find(parent, parent[node])  #recursively find the root
    return parent[node]

# "union" function for the union-find algorithm
def union(parent, rank, node1, node2):
    #combines two sets (represented by their roots) into one
    root1 = find(parent, node1)
    root2 = find(parent, node2)

    # attach the smaller tree under the larger tree
    if root1 != root2:
        if rank[root1] > rank[root2]:
            parent[root2] = root1
        elif rank[root1] < rank[root2]:
            parent[root1] = root2
        else:
            parent[root2] = root1
            rank[root1] += 1  #increase rank if ranks are equal

#Kruskal's Algorithm to find the Minimum Spanning Tree (MST)
def kruskal(nodes, edges):
    mst = []  #store the edges of the MST
    total_weight = 0  #initialize total weight of the MST

    #initialize parent and rank dictionaries for union-find
    parent = {node: node for node in nodes}  #each node is its own parent initially
    rank = {node: 0 for node in nodes}  #initialize rank of each node to 0

    #sort edges by weight
    edges.sort(key=lambda x: x[2])

    for edge in edges:
        u, v, weight = edge  #get source, destination, and weight of the edge

        #check if adding the edge creates a cycle
        if find(parent, u) != find(parent, v):
            #ff no cycle, add the edge to the MST
            union(parent, rank, u, v)
            mst.append((u, v, weight))
            total_weight += weight  #update the total weight of the MST

    return mst, total_weight

#get all unique nodes from the dataset
nodes = set(df['source']).union(set(df['destination']))

#run Kruskal's algorithm
mst_edges, total_weight = kruskal(nodes, edges)

# Print resuts
print("Total Weight of the Minimum Spanning Tree:", total_weight)
print("Edges in the Minimum Spanning Tree:")
for edge in mst_edges:
    print(f"{edge[0]} -> {edge[1]} with weight {edge[2]}")

# Save to CSV file
output_file = "QMST_edges.csv" 
mst_df = pd.DataFrame(mst_edges, columns=['source', 'destination', 'weight'])  #convert MST edges to a DataFrame
mst_df.to_csv(output_file, index=False)  #save the DataFrame to a CSV file

mst_graph = nx.Graph()
for u, v, weight in mst_edges:
    #add edges to the graph with their weights as attributes
    mst_graph.add_edge(u, v, weight=weight)

plt.figure(figsize=(16, 12))  #set the size of the visualization canvas

# Use for positioning the nodes
pos = nx.spring_layout(mst_graph, k=1.0, iterations=200)

#draw the graph nodes and edges
nx.draw(
    mst_graph,
    pos,  #positions of nodes
    with_labels=True,  #display node labels
    node_color="lightblue",  #set node color
    edge_color="gray",  #set edge color
    node_size=2000,  #set node size
    font_size=10,  #set font size for labels
)

# add edge labels
edge_labels = {(u, v): f"{d['weight']}" for u, v, d in mst_graph.edges(data=True)}
nx.draw_networkx_edge_labels(
    mst_graph,
    pos, 
    edge_labels=edge_labels,  
    font_color="red",
    font_size=10,
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7)  
)

#adjust the layout
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

plt.title("Quantum Minimum Spanning Tree (QMST)")

#show it
plt.show()


"""
This programm implements Kruskal's algorithm to find the Minimum Spanning Tree (MST) of a graph. 
It starts by loading graph data from a CSV file, with edges representing connections between nodes, and then constructs 
the MST using a union-find data structure to avoid cycles. The algorithm sorts the edges by weight and processes them in 
ascending order, adding each edge to the MST if it does not form a cycle. The resulting MST is saved to a new CSV file and
visualized using networkx and matplotlib, where nodes and edges are displayed with their respective weights. 

Time complexity - O(E log E) where E is the number of edges
Memory complexity - O(E + N) where N is the number of nodes
"""