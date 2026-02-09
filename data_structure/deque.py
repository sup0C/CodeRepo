class Deque:
    '''
    List下标0作为deque的尾端; List下标-1作为deque的首端
    操作复杂度：addFront或removeFront为O(1)；addRear或removeRear为O(n)
    '''
    def __init__(self):
        self.items = []

    def isEmpty(self) :
        return self.items == []

    def addFront(self, item) :
        self.items .append(item)

    def addRear(self,item):
        self.items.insert(0,item)

    def removeFront(self):
        return self.items .pop()

    def removeRear(self):
        return self.items .pop(0)

    def size(self):
        return len(self.items)