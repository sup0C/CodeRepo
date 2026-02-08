from graphs import Graph, Vertex
from priorityQueue import PriorityQueue
import sys

def prim(aGraph,start):
    pq = PriorityQueue()
    for v in G:
        v.setDistance(sys.maxsize) # 先将所有顶点的disc设为最大
        v.setPred(None)
    start.setDistance(0)
    pq.buildHeap([(v.getDistance(),v)for v in G])
    while not pq.isEmpty():
        currentVert = pq.delMin() #
        for nextVert in currentVert.getConnections():
            newCost = currentVert.getWeight(nextVert)
            if nextVert in pq and newCost < nextVert.getDistance():
                nextVert.setPred(currentVert)
                nextVert.setDistance(newCost)
                pq.decreaseKey(nextVert,newCost)