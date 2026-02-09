class Queueinit:
    '''
    将List首端作为队列尾端，List的末端作为队列首端，若首尾倒过来实现，则复杂度也倒过来。
    enqueue()复杂度为0(n)；dequeue()复杂度为0(1)
    '''
    def __init__(self):
        self.items = []
    def isEmpty(self):
        return self.items == []
    def enqueue(self,item):
        self.items.insert(0,item)
    def dequeue(self):
        return self.items .pop( )
    def size(self):
        return len(self.items)