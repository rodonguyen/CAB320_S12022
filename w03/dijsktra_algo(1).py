'''

Demonstration of the networkx library

Implementation of Dijsktra Algorithm to compute the shortest path
from a distinguished vertex to all other vertices.

last modified 30/01/2022
by f.maire@qut.edu.au

'''


# for drawing graphs
import matplotlib.pyplot as plt

# graph representation
import networkx as nx

from math import inf




def make_graph_expl_1():
    '''
    Create a graph G with weighted edges

    Returns
    -------
    G : Graph
        Some toy example of a weighted graph.
    '''
    
    G = nx.Graph() # empty graph
    
    G.add_nodes_from(range(9)) # add nodes
    
    # add edges
    G.add_weighted_edges_from(
        [(0,1,4.0),(0,7,8.0),(1,7,11.0),(1,2,8.0),(7,7,8.0),
         (7,6,1.0),(6,8,6.0),(2,8,2.0),(2,5,4.0),(2,3,7.0),
         (6,5,2.0),(3,5,14.0),(3,4,9.0),(5,4,10.0)
         ])
    
    # specifiy position of the nodes
    pos = [(0,1), # 0
           (1,2), # 1
           (2,2), # 2
           (3,2), # 3
           (4,1), # 4
           (3,0), # 5
           (2,0), # 6
           (1,0), # 7
           (2,1) # 8
           ]
    # G['node_pos'] = pos
    G.node_pos = pos
    return G


def Dijkstra_shortest_path(G, start):
    '''    
    Implementation of Dijsktra algorithm
    https://en.wikipedia.org/wiki/Dijkstra
    Compute the shortest paths from node start to all the other nodes in G.    
    Return D, P   where D[v]  is the cost of the cheapest path from start to 
                  node v,  and P[v] is the parent node of v on this optimal 
                  path from start to v.
    '''
    def get_closest():
        '''
        Find in L, the element u with the smallest dist[u]
        Remove u from L and return the pair   u, dist[u]
        '''
        
        u = L[0]
        du = dist[u]
        for v in L[1:]:
            if dist[v]<du:
                u,du = v, dist[v]
        L.remove(u)
        return u, du
    # .........................................................................
    
    
    L = list(G.nodes) # List of nodes that have not been finalized
    
    # A more efficient implementation would use a priority queue for L,
    # but for the sake of simplicity we use a standard list
    
    dist = {v:inf for v in G.nodes} # mapping v -> cost of best path 
                                    # known so far from node start to v
    parent = {v:None for v in G.nodes} # mapping v -> parent in the tree of 
                                       # shortest paths
    dist[start] = 0

    while L: # while there are unfinalized nodes
        u, du = get_closest() #
        for v in G.neighbors(u):
            if du+G.adj[u][v]['weight'] < dist[v]:
                dist[v] = du+G.adj[u][v]['weight']
                parent[v] = u
    
    return dist, parent
    # -------------------------------------------------------------------------

if __name__ == '__main__':
    
    G = make_graph_expl_1() # Create a graph
    # print(G.nodes())
    
    # Add weights to the edges
    w_labels = {e:G.edges[e]['weight'] for e in G.edges}

    # Draw the graph
    nx.draw(G, G.node_pos, with_labels=True)
    nx.draw_networkx_edge_labels(G, G.node_pos, edge_labels=w_labels)    
    plt.show() # show the plot
    
    # Compute the shortest paths from Vertex 0 to all the other vertices
    # D[v] is the cost of going from vertex 0 to vertex v
    # All the shortest paths belong to a tree rooted at 0
    # This tree is represented with the mapping P
    # P[v] is the parent of v in the tree.
    D, P = Dijkstra_shortest_path(G, 0)
    
    print(f'Mapping of Cost:\n{D=}\n')
    
    print(f'Parent mapping:\n{P=}\n')





# ++++++++++++++++++++++    CODE CEMETARY, please ignore!    ++++++++++++++++++++++
        

    #     heap = []            # creates an empty heap
    # heappush(heap, item) # pushes a new item on the heap
    # item = heappop(heap) # pops the smallest item from the heap
    # item = heap[0]       # smallest item on the heap without popping it
    # heapify(x)           # transforms list into a heap, in-place, in linear time
# Create an empty graph with no nodes and no edges
# G = nx.Graph()

# G.add_nodes_from([
#     (1, {"color": "red"}),
#     (2, {"color": "red"}),
#     (3, {"color": "red"}),
#     (4, {"color": "red"}),
#     (5, {"color": "green"}),
#     (6, {"color": "green"})])

# Add a list of edges. Nodes are created if needed
# G.add_edges_from([(1, 2), (2, 3),(3,1),(1,4), (5,6)])

# G.add_weighted_edges_from([(1,2,1.0),(1,3,5.0),(2,3,1.0),(4,1,3.0),(5,6,2.0)])

# nx.draw(G, with_labels=True)

# plt.show()
# C = [ c for x,c in G.nodes('color')]

# nx.draw(G,node_color=C, with_labels=True)



