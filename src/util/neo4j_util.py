# encoding=utf-8
from py2neo import Graph, Node, Relationship


class Neo4jClient:

    def __init__(self, host, port, username, password):
        self.graph = Graph(host + ':' + str(port), username=username, password=password)

    def create_node(self, type, name):
        node = Node(type, name=name)
        self.graph.create(node)

    def create_relationship(self, node1, relation, node2):
        node1_relation_node2 = Relationship(node1, relation, node2)
        self.graph.create(node1_relation_node2)


if __name__ == '__main__':
    nc = Neo4jClient('http://127.0.0.1', 7474, 'neo4j', '123456')
    # nc.create_node('Unit', '1号机组')
    # nc.create_node('PowerPlant', '福州电厂')
    node1 = Node('Unit', name='2号机组')
    print(node1)
    node1 = nc.graph.find_one('Unit', property_key='name', property_value='1号机组')
    # node2 = Node('PowerPlant', name='福州电厂')
    # nc.create_relationship(node1, '属于', node2)
    node = nc.graph.run("match(n:Unit{name:'2号机组'}) return n")
    for n in node:
        for key in n.get('n'):
            print(n.get('n').get(key))