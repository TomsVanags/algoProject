import pandas as pd  
import networkx as nx 
import matplotlib.pyplot as plt 

#reads CSV file and creates an undirected graph
def read_graph(file_path):
    df = pd.read_csv(file_path) #pandas to read the CSV file into a DataFrame
    G = nx.Graph() #create an undirected graph object

    #iterate over each row in the DataFrame and add an edge between the source and destination
    for _, row in df.iterrows():
        G.add_edge(row['source'], row['destination'], weight=row['distanceLY'])
    
    return G #returns the graph

#finds all cycles in the graph using DFS
def find_cycles(graph):
    cycles = [] #store the cycles
    visited = set() #track visited nodes during DFS
    stack = [] #stack to keep track of the nodes in the current DFS path

    #helper function to perform DFS traversal
    def dfs(node, parent):
        visited.add(node) #mark the current node as visited
        stack.append(node) #add the current node to the stack

        #iterate over all the neighbors of the current node
        for neighbor in graph.neighbors(node):
            if neighbor == parent:
                continue #skip the parent node to avoid traversing the same edge
            if neighbor in visited:
                #if a cycle is found get the cycle
                if neighbor in stack:
                    cycle = stack[stack.index(neighbor):] + [neighbor] #get the cycle from the stack
                    cycles.append(cycle) #add the cycle to the list of cycles
            else:
                dfs(neighbor, node) #recursively explore the neighbor

        stack.pop() #backtrack by removing the current node from the stack

    #perform DFS for each unvisited node in the graph
    for node in graph.nodes():
        if node not in visited:
            dfs(node, None) #start DFS traversal for the node

    return cycles #return the list of cycles found in the graph

#calculates the total weight of a cycle by summing the weights of the edges in the cycle
def calculate_cycle_weight(graph, cycle):
    weight = 0  # Initialize the total weight of the cycle

    for i in range(len(cycle)):
        u = cycle[i] #get the current node in the cycle
        v = cycle[(i + 1) % len(cycle)] #get the next node in the cycle

        #check if the edge exists in the graph and sum its weight
        if graph.has_edge(u, v):
            weight += graph[u][v]['weight']
        elif graph.has_edge(v, u): #if the edge is undirected, check the reverse direction
            weight += graph[v][u]['weight']

    return weight #returns weight of the cycle

#finds the cycle with the maximum weight from a list of cycles
def find_longest_loop(graph, cycles):
    max_weight = 0 #set weight to 0 at first
    longest_cycle = None #and this as none, so both can be increased

    #iterate over each cycle to find the one with the largest weight
    for cycle in cycles:
        weight = calculate_cycle_weight(graph, cycle) #calculate the weight of the cycle
        if weight > max_weight: #if this cycle is larger, update max_weight and longest_cycle
            max_weight = weight
            longest_cycle = cycle

    return longest_cycle, max_weight #return the longest things

#visualizes the given cycle in the graph using Matplotlib
def visualize_cycle(graph, cycle, title):
    subgraph = graph.subgraph(cycle) #create a subgraph containing only the nodes in the cycle
    pos = nx.spring_layout(subgraph) #generate the positions of the nodes for visualization

    plt.figure(figsize=(8, 6)) #set the figure size
    nx.draw(subgraph, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=12) #draw the subgraph

    #create labels for the edges showing the weight of each edge
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in subgraph.edges(data=True)}
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_color='red') #draw the edge labels

    plt.title(title)
    plt.show() #show it

def main():
    file_path = "../actual_distances_space_graph.csv" 

    graph = read_graph(file_path) #red the csv file

    cycles = find_cycles(graph) #finds all cycles

    longest_cycle, max_weight = find_longest_loop(graph, cycles) #find the longest cycle and max weight

    print("Longest Cycle:", longest_cycle) 
    print("Sum of Weights in Longest Loop:", max_weight) 

    if longest_cycle: #if a cycle was found, visualize it
        visualize_cycle(graph, longest_cycle, "Longest Cycle in the Space-Time Fabric")
    else: # If no cycle was found
        print("No cycles found in the graph.")

if __name__ == "__main__":
    main()

"""
The code analyzes a graph constructed from a CSV file, looking for cycles and identifying 
the longest cycle in terms of edge weight. It reads the graph data, then uses a Depth-First Search (DFS) 
approach to find all cycles. The weight of each cycle is computed by summing the weights of the 
edges involved, and the cycle with the highest weight is identified. If a cycle is found, it is visualized 
with node and edge labels using Matplotlib.

Time complexity - O(V + E + C) where E is the number of edges, V is number of nodes, C is number of cycles
Memory complexity - O(V + E + C)
"""