# encoding=utf-8
from py2neo import Graph, Node, Relationship


class Neo4jClient:

    def __init__(self, config, logger):
        self.logger = logger
        self.graph = self._create_graph(config)

    def _create_graph(self, config):
        try:
            return Graph(config['neo4j_host'] + ':' + str(config['neo4j_port']),
                  username=config['neo4j_username'], password=config['neo4j_password'])
        except Exception as e:
            self.logger.error(e)

    def create_node(self, type, name):
        is_success = True
        try:
            node = Node(type, name=name)
            self.graph.create(node)
        except Exception as e:
            self.logger.error(e)
            is_success = False
        return is_success

    def create_relationship(self, node1, relation, node2):
        is_success = True
        try:
            node1_relation_node2 = Relationship(node1, relation, node2)
            self.graph.create(node1_relation_node2)
        except Exception as e:
            self.logger.error(e)
            is_success = False
        return is_success


if __name__ == '__main__':
    config = {
        'neo4j_host': 'http://127.0.0.1',
        'neo4j_port': 7474,
        'neo4j_username': 'neo4j',
        'neo4j_password': 'moon0816'
    }
    import logging
    nc = Neo4jClient(config, logging)
    # nodes = nc.graph.find('Person', property_key='name', property_value='燕双鹰')
    # for node in nodes:
    #     print(node)
    # nc.create_node('Unit', '1号机组')
    # nc.create_node('PowerPlant', '福州电厂')
    # node1 = Node('Unit', name='2号机组')
    # node2 = Node('PowerPlant', name='福州电厂')
    # nc.create_relationship(node1, '属于', node2)
    # node = nc.graph.run("match(n:Unit{name:'2号机组'}) return n")
    # for n in node:
    #     for key in n.get('n'):
    #         print(n.get('n').get(key))
    nodes = nc.graph.run("match(n:Person {name: '燕双鹰', age: '44'}) return n", ).data()
    print(type(nodes[0].get('n')))