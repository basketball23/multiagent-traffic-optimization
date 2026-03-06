import sumolib

net = sumolib.net.readNet('grid-network.net.xml')

edge_ids = [edge.getID() for edge in net.getEdges()]

print(edge_ids)