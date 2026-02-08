class BinaryTree:
    def __init__(self,rootObj):
        self.key = rootObj # 成员key保存根节点数据项
        # 成员left/rightChild则保存指向左/右子树的引用（同样是BinaryTree对象）
        self.leftChild = None
        self.rightChild = None

    def insertLeft(self,newNode):
        if self.leftChild == None:
            self.leftChild = BinaryTree(newNode)
        else:
            # 当插入位置有数据时，将新节点插入树，作为其直接的左子节点
            t = BinaryTree(newNode)
            t.leftChild = self.leftChild
            self.leftChild = t

    def insertRight(self,newNode):
        if self.rightChild == None:
            self.rightChild = BinaryTree(newNode)
        else:
            # 当插入位置有数据时，将新节点插入树，作为其直接的右子节点
            t = BinaryTree(newNode)
            t.rightChild = self.rightChild
            self.rightChild = t


    def getRightChild(self):
        return self.rightChild

    def getLeftChild(self):
        return self.leftChild

    def setRootVal(self,obj):
        self.key = obj

    def getRootVal(self):
        return self.key

    def preorder(self):
        # 前序遍历，从上（根部）往下（叶部）遍历
        print(self.key)
        if self.leftChild:
            self.leftChild.preorder()
        if self.rightChild:
            self.rightChild.preorder()

    def inorder(self):
        # 中序遍历
        if self.leftChild:
            self.leftChild.inorder()
        print(self.key)
        if self.rightChild:
            self.rightChild.inorder()

    def postorder(self):
        # 后序遍历，从下（叶部）往上（根部）遍历
        if self.leftChild:
            self.leftChild.postorder()
        if self.rightChild:
            self.rightChild.postorder()
        print(self.key)

    def printexp(self):
        # 将二叉树（子树的节点为操作数，树叶为操作数）解析为全符号表达式
        if self.leftChild:
            print('(', end=' ')
            self.leftChild.printexp()
        print(self.key, end=' ')
        if self.rightChild:
            self.rightChild.printexp()
            print(')', end=' ')

def printexp(tree):
    # 将二叉树（子树的节点为操作数，树叶为操作数）解析为全符号表达式
    sVal = ""
    if tree:
        sVal = '(' + printexp(tree.getLeftChild())
        sVal = sVal + str(tree.getRootVal())
        sVal = sVal + printexp(tree.getRightChild()) + ')'
    return sVal



if __name__ == '__main__':
    r = BinaryTree('a')
    print(r.getRootVal());print(r.getLeftChild())
    r.insertLeft('b')
    print(r.getLeftChild());print(r.getLeftChild().getRootVal())
    r.insertRight('c')
    print(r.getRightChild());print(r.getRightChild().getRootVal())
    r.getRightChild().setRootVal('hello')
    print(r.getRightChild().getRootVal())
    # 将将二叉树（子树的节点为操作数，树叶为操作数）转换为全符号表达式
    x = BinaryTree('*');x.insertLeft('+')
    l = x.getLeftChild();l.insertLeft(4);l.insertRight(5);x.insertRight(7)
    print(x.printexp());print(printexp(x))
