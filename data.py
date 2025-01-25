import networkx as nx

class Instance:
    def __init__(self, trains, objectives):
        self.trains = trains
        self.objectives = objectives
        self.trains_graphs = []
        self.deadlock_graph = nx.DiGraph
        self.create_train_graphs()


    def create_train_graphs(self):
        for train in self.trains:
            graph = nx.DiGraph()
            graph.add_nodes_from([i for i, _ in enumerate(train)])

            for i, operation in enumerate(train):
                graph.add_edges_from([(i, v) for v in operation["successors"]])
            self.trains_graphs.append(graph)


class Solution:
    def __init__(self, objective_value, events):
        self.objective_value = objective_value
        self.events = events

