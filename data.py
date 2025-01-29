import networkx as nx

class Instance:
    def __init__(self, trains : list, objectives):
        self.trains = trains
        self.objectives = objectives

        # Only calculate if requested for efficiency
        self.__train_graphs = None
        self.__resource_conflicts = None
        self.__deadlock_graph = None
        self.__deadlocks = None


    def get_train_graphs(self):
        if not self.__train_graphs:
            self.__train_graphs = self.create_train_graphs()


    def get_resource_conflicts(self):
        if not self.__resource_conflicts:
            self.__resource_conflicts = self.create_resource_conflict_mapping()
        return self.__resource_conflicts


    def get_deadlocks(self):
        if not self.__deadlock_graph:
            self.__deadlock_graph = self.get_deadlock_graph()
        if not self.__deadlocks:
            self.__deadlocks = list(nx.simple_cycles(self.__deadlock_graph))
        return self.__deadlocks


    def get_deadlock_graph(self):
        if not self.__deadlock_graph:
            self.__deadlock_graph = self.create_deadlock_graph()
        return self.__deadlock_graph


    def create_train_graphs(self):
        train_graphs = []
        for train in self.trains:
            graph = nx.DiGraph()
            graph.add_nodes_from([i for i, _ in enumerate(train)])

            for i, operation in enumerate(train):
                graph.add_edges_from([(i, v) for v in operation["successors"]])
            train_graphs.append(graph)
        return train_graphs


    def create_resource_conflict_mapping(self):
        resource_conflicts = dict()
        for t, train in enumerate(self.trains):
            # Create a mapping that maps a ressource to the list of operations using that ressource
            for o, op in enumerate(train):
                for res in op["resources"]:

                    if res["resource"] in resource_conflicts.keys():
                        resource_conflicts[res["resource"]].append((t, o))
                    else:
                        resource_conflicts[res["resource"]] = [(t, o)]
        return resource_conflicts


    def create_deadlock_graph(self):
        graph = nx.DiGraph()
        for i, train in enumerate(self.trains):
            for j, op in enumerate(train):
                for res in op["resources"]:

                    if type(res["release_time"]) == 0:
                        continue

                    for succ in op["successors"]:
                        for succ_res in train[succ]["resources"]:
                            edge = (res["resource"], succ_res["resource"])

                            if edge[0] == edge[1]:
                                continue
                            if edge in graph.edges:
                                edge_data = graph[edge[0]][edge[1]].get("data", [])
                                edge_data.append((i, j))
                                graph[edge[0]][edge[1]]["data"] = edge_data
                            else:
                                graph.add_nodes_from([res["resource"], succ_res["resource"]])
                                graph.add_edge(edge[0], edge[1], data=[(i, j)])
        return graph

