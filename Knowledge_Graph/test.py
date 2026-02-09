from py2neo import Graph
# Py2neo 的核心是 Graph 类，它代表一个图数据库连接对象，负责管理节点和关系的增删改查。
# 1 连接
graph = Graph("bolt://localhost:7687", auth=("neo4j", "123qwe!@"))
graph.delete_all()


# 2 创建节点和关系
from py2neo import Node, Relationship
alice = Node("Person", name="Alice", age=30)
bob = Node("Person", name="Bob")
carol = Node("Person", name="Carol")
graph.create(alice | bob | carol)

friendship = Relationship(alice, "KNOWS", bob)
alice["email"] = "alice@example.com" # 属性操作
graph.create(friendship)
graph.create(Relationship(bob, "KNOWS", carol))
graph.create(Relationship(alice, "KNOWS", carol))

# 3 查询节点 - 使用 run 执行 Cypher 查询
# 查询所有人名
result = graph.run("MATCH (p:Person) RETURN p.name AS name").data()
for r in result:
    print(r["name"])

# 4 条件查询与过滤
# 查询alice的朋友
result = graph.run("""
    MATCH (a:Person)-[:KNOWS]->(b:Person)
    WHERE a.name = $name
    RETURN b.name AS friend
""", name="Alice").data()

print("Alice's friends:", [r["friend"] for r in result])