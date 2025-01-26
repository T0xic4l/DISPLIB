import networkx as nx
import itertools

class EventSorter:
    def __init__(self, feasible_sol):
        self.feasible_sol = feasible_sol
        self.events = self.create_events(feasible_sol)
        self.topological_sort()


    def topological_sort(self):
        sorted_events = []
        lhs, rhs = 0, 0

        while lhs < len(self.events):
            # Find the intervall of events that take place at the same time
            while rhs < len(self.events) and self.events[rhs]["time"] == self.events[lhs]["time"]:
                rhs += 1

            # If there are at least two events at the same time, we have to make sure they are sorted correctly
            if rhs - lhs > 1:
                topological_graph = self.create_topological_graph(self.events[lhs:rhs])

                for component in list(nx.weakly_connected_components(topological_graph)):
                    # Create that subgraph
                    subgraph = self.create_subgraph(topological_graph, component)

                    # Now, topologically sort it
                    for node in list(nx.topological_sort(subgraph)):
                        # reorder the group of events and add it to the sorted list
                        sorted_events.append(self.events[lhs + node])
            else:
                sorted_events.append(self.events[lhs])

            # update the starting index to find the next group of events
            lhs = rhs
        self.events = sorted_events


    def create_topological_graph(self, events):
        # create a graph that represents the dependencies between operations: Graph may not be connected !!!
        graph = nx.DiGraph()
        graph.add_nodes_from(range(len(events)))

        # Add predecessor-edges
        for i, event in enumerate(events):
            # Retrieve the resources that need to be freed, so that the given operation can happen
            needed_resources = [res["resource"] for res in self.feasible_sol[event["train"]][event["operation"]]["resources"]]

            # Get the indices of events that need to take place before, so their predecessors resources are freed
            for j, other in enumerate(events):
                predecessor = self.get_predecessor(other["train"], other["operation"])

                # Of course an event is self-dependent. Skip that case to avoid self-loops. Otherwise, sorting will not work!
                if i == j or predecessor is None or event["train"] == other["train"]:
                    continue

                predecessor_resources = [res["resource"] for res in
                                         self.feasible_sol[other["train"]][predecessor]["resources"]]

                # If a predecessor of one of these events that took a needed resource, make sure the event takes place before the other event
                for res in needed_resources:
                    if res in predecessor_resources:
                        graph.add_edge(j, i)
                        break

        # Add priority-edges for same resources
        for i, event in enumerate(events):
            '''
            WE CHANGED SOMETHING HERE: END - END -> END - START
            '''
            duration = self.feasible_sol[event["train"]][event["operation"]]["end"] - self.feasible_sol[event["train"]][event["operation"]]["start"]

            critical_resources = []

            if duration > 0:
                # If the duration is non-zero, every operation that uses one of that resources needs to happen before it
                critical_resources.append(
                    r["resource"] for r in self.feasible_sol[event["train"]][event["operation"]]["resources"])
            else:
                for r in self.feasible_sol[event["train"]][event["operation"]]["resources"]:
                    # Else, only care about the resources with a non-zero release_time
                    if r["release_time"] > 0:
                        critical_resources.append(r["resource"])

            for j, other in enumerate(events):
                if i == j or event["train"] == other["train"]:
                    continue

                # find other events j with same resource
                for other_r in self.feasible_sol[other["train"]][other["operation"]]["resources"]:
                    if other_r["resource"] in critical_resources:
                        graph.add_edge(j, i)
                        break

        # Add chronological-edges
        train_to_events = dict()
        for i, event in enumerate(events):
            if event["train"] in train_to_events.keys():
                train_to_events[event["train"]].append(i)
            else:
                train_to_events[event["train"]] = [i]

        for train, train_events in train_to_events.items():
            if len(train_events) > 1:
                # If the same train has two events at the same time, ensure that operation n+1 happens AFTER n
                for index1, index2 in itertools.combinations(train_events, 2):
                    if events[index1]["operation"] > events[index2]["operation"]:
                        graph.add_edge(index2, index1)
                    else:
                        graph.add_edge(index1, index2)

        return graph


    def get_predecessor(self, train, op):
        if op == 0:
            return None

        operations = list(self.feasible_sol[train].keys())
        for i, to_find in enumerate(operations):
            if to_find == op:
                return operations[i - 1]


    @staticmethod
    def create_events(feasible_sol):
        events = []

        for i, train in enumerate(feasible_sol):
            for op, timings in train.items():
                events.append({"time": timings["start"], "train": i, "operation": op})

        return sorted(events, key=lambda x: x["time"])


    @staticmethod
    def create_subgraph(graph: nx.DiGraph, nodes):
        subgraph = nx.DiGraph()
        subgraph.add_nodes_from(nodes)
        subgraph.add_edges_from([edge for edge in graph.edges if edge[0] in nodes and edge[1] in nodes])
        return subgraph