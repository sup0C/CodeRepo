class Node:
    '''
    链表实现的最基本元素是节点Node，每个节点至少要包含2个信息：数据项本身，
    以及指向下一个节点的引用信息。注意：next为None的意义是没有下一个节点了， 这个很重要。
    '''
    def __init__(self,initdata):
        self.data = initdata
        self.next = None

    def getData(self):
        return self.data

    def getNext(self):
        return self.next

    def setData(self,newdata):
        self.data = newdata

    def setNext(self,newnext):
        self.next = newnext
