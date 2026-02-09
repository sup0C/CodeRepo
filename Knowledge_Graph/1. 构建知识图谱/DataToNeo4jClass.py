# -*- coding: utf-8 -*-
from py2neo import Node, Graph, Relationship, NodeMatcher

class DataToNeo4j(object):
    """将excel中数据存入neo4j"""
    def __init__(self):
        """建立连接"""
        # link = Graph(r"http://localhost:7687", auth=("neo4j", "123qwe!@")) # 会报错
        # link = Graph(scheme="bolt", host="localhost", port=7687,secure=True,auth=("neo4j", "123qwe!@"))
        link = Graph("bolt://localhost:7687", auth=("neo4j", "123qwe!@"))
        # link=Graph('http://localhost:7474', username='neo4j', password='123qwe!@') # 老版本
        self.graph = link
        # self.graph = NodeMatcher(link)
        # 定义label，定义标签
        self.buy = 'buy'  # 购买方
        self.sell = 'sell'  # 销售方
        self.graph.delete_all()  # 删除已有的节点和关系、清空
        # NodeMatcher是从py2neo中导入的    后续帮助做匹配
        self.matcher = NodeMatcher(link)  # 定义一个matcher，一会定义关系的时候要用

        # 下边注释掉的是一些官方的小例子，做测试的时候可以试一试
        ##Node是从py2neo中导入的
        """
        #创建节点
        node3 = Node('animal' , name = 'cat')
        node4 = Node('animal' , name = 'dog')  
        node2 = Node('Person' , name = 'Alice')
        node1 = Node('Person' , name = 'Bob')  
        #创建关系、边
        r1 = Relationship(node2 , 'know' , node1)    
        r2 = Relationship(node1 , 'know' , node3) 
        r3 = Relationship(node2 , 'has' , node3) 
        r4 = Relationship(node4 , 'has' , node2) 
        #create就是实际的添加到图当中   
        self.graph.create(node1)
        self.graph.create(node2)
        self.graph.create(node3)
        self.graph.create(node4)
        self.graph.create(r1)
        self.graph.create(r2)
        self.graph.create(r3)
        self.graph.create(r4)
        """

    def create_node(self, node_buy_key, node_sell_key):
        """建立节点
        输入是去重的买方、卖方
        """
        for name in node_buy_key:
            buy_node = Node(self.buy, name=name)  # 第一个参数是标签，第二个参数是名字
            self.graph.create(buy_node)
        for name in node_sell_key:
            sell_node = Node(self.sell, name=name)
            self.graph.create(sell_node)

    def create_relation(self, df_data):
        """建立联系
        输入是一个词典，包含买方、卖方和金额三方的数据
        """
        m = 0
        for m in range(0, len(df_data)):
            # 遍历数据中的每一条数据
            try:
                # 寻找当前买方和卖方的节点实体
                print(list(self.matcher.match(self.buy).where("_.name=" + "'" + df_data['buy'][m] + "'")))
                print(list(self.matcher.match(self.sell).where("_.name=" + "'" + df_data['sell'][m] + "'")))
                # 建立两个节点间的关系
                rel = Relationship(
                    self.matcher.match(self.buy).where("_.name=" + "'" + df_data['buy'][m] + "'").first(),
                    df_data['money'][m],
                    self.matcher.match(self.sell).where("_.name=" + "'" + df_data['sell'][m] + "'").first())

                self.graph.create(rel) # 建立关系
            except AttributeError as e:
                print(e, m)