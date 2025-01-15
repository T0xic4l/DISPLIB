import argparse, json, itertools

import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from ortools.sat.python import cp_model as cp


class Instance:
    def __init__(self, trains, objectives):
        self.trains = trains
        self.objectives = objectives


class Solution:
    def __init__(self, objective_value, events):
        self.objective_value = objective_value
        self.events = events


class DisplibSolver:
    def __init__(self, instance):
        self.trains = instance.trains
        self.objectives = instance.objectives
        self.train_graphs = [create_train_graph(train) for train in self.trains]
        self.model = cp.CpModel()
        self.solver = cp.CpSolver()

        self.trains_per_res = dict() # A mapping from a resource to a list of operations using this operation
        self.op_start_vars = dict()
        self.op_end_vars = dict()
        self.edge_select_vars = list()
        self.threshold_vars = dict()

        for i, train in enumerate(self.trains):
            select_vars = {}

            for j, op in enumerate(train):
                self.op_start_vars[i,j] = self.model.NewIntVar(lb=op["start_lb"], ub=op["start_ub"], name=f"Start of Train {i} : Operation {j}")
                self.op_end_vars[i,j] = self.model.NewIntVar(lb=op["start_lb"] + op["min_duration"], ub=2**20, name=f"End of Train {i} : Operation {j}")

                # Create a mapping that maps a ressource to the list of operations using that ressource
                for res in op["resources"]:
                    if res["release_time"] == 0:
                        res["release_time"] = self.model.NewBoolVar(name=f"rt for {(i, j)} : res {res}")

                    if res["resource"] in self.trains_per_res.keys():
                        self.trains_per_res[res["resource"]].append((i,j))
                    else:
                        self.trains_per_res[res["resource"]] = [(i,j)]

                for s in op["successors"]:
                    select_vars[j,s] = self.model.NewBoolVar(name=f"Train {i} : Edge<{j},{s}>")
            self.edge_select_vars.append(select_vars)

            for obj in self.objectives:
                if obj["coeff"] != 0:
                    self.threshold_vars[obj["train"], obj["operation"]] = self.model.NewIntVar(lb=0, ub=2**20, name=f"Threshold of Train {obj["train"]} : Operation {obj["operation"]}")
                else:
                    self.threshold_vars[obj["train"], obj["operation"]] = self.model.NewBoolVar(name=f"Threshold of Train {obj["train"]} : Operation {obj["operation"]}")

        self.res_graph = create_deadlock_graph(self.trains, self.trains_per_res.keys())

        self.add_threshold_constraints()
        self.add_path_constraints()
        self.add_timing_constraints()
        self.add_resource_constraints()
        self.add_deadlock_constraints()

        self.set_objective()


    def add_threshold_constraints(self):
        for i, obj in tqdm(enumerate(self.objectives), desc="Adding threshold-constraints"):
            train = obj["train"]
            op = obj["operation"]

            # There is no delay if start = threshold in case coeff != 0
            if obj["coeff"] != 0:
                self.model.add(self.threshold_vars[obj["train"], obj["operation"]] >= self.op_start_vars[train, op] - obj["threshold"])
            else:
                # But there is a delay if start = threshold if increment != 0 (check out the objective-formular in 2.1.1)
                self.model.add(self.op_start_vars[train, op] + 1 <= obj["threshold"]).OnlyEnforceIf(self.threshold_vars[train, op].Not())


    def add_path_constraints(self):
        for i, train in tqdm(enumerate(self.trains), desc="Adding path-constraints"):
            last_op = len(self.train_graphs[i].nodes) - 1

            # For the first operation, exactly one outgoing edge must be chosen
            # For the last operation, exactly one ingoing edge must be chosen
            self.model.add(sum(self.edge_select_vars[i][out_edge] for out_edge in self.train_graphs[i].out_edges(0)) == 1)
            self.model.add(sum(self.edge_select_vars[i][in_edge] for in_edge in self.train_graphs[i].in_edges(last_op)) == 1)

            # For all operations inbetween, the number of chosen outgoing and ingoing edges must be equal
            for j in range(1, last_op):
                in_edges = self.train_graphs[i].in_edges(j)
                out_edges = self.train_graphs[i].out_edges(j)
                self.model.add(sum(self.edge_select_vars[i][in_edge] for in_edge in in_edges) ==
                               sum(self.edge_select_vars[i][out_edge] for out_edge in out_edges))


    def add_timing_constraints(self):
        # Guarantee that every operation cannot start before the chosen predecessor (start + min_duration as a lower bound)
        for i, train in tqdm(enumerate(self.trains), desc="Adding timing-constraints"):
            for op in range(len(self.train_graphs[i].nodes)):
                for in_edge in self.train_graphs[i].in_edges(op):
                    # If operation is chosen, successor may only start after at least start_var + min_duration
                    self.model.add(self.op_start_vars[i, in_edge[0]] + self.trains[i][in_edge[0]]["min_duration"] <= self.op_start_vars[i, op]).OnlyEnforceIf(self.edge_select_vars[i][in_edge])

                for out_edge in self.train_graphs[i].out_edges(op):
                    # Operation ends when successor operation starts
                    self.model.add(self.op_end_vars[i, op] == self.op_start_vars[i, out_edge[1]]).OnlyEnforceIf(self.edge_select_vars[i][out_edge])


    def add_resource_constraints(self):
        for i, (res, ops) in enumerate(tqdm(self.trains_per_res.items(), desc="Adding resource-constraints")):
            # If there are multiple operations that use the same resource, a conflict could – in theory – be possible
            if len(ops) > 1:
                for (train_1, op_1), (train_2, op_2) in itertools.combinations(ops, 2):
                    # Since operations per train do not overlap due to the timing constraints, we can skip this case
                    if train_1 == train_2:
                        continue

                    # Boolvars for overlapping constraints
                    op_1_chosen = self.model.NewBoolVar(name=f"Train {train_1} : Operation {op_1} is chosen")
                    op_2_chosen = self.model.NewBoolVar(name=f"Train {train_2} : Operation {op_2} is chosen")

                    # The first and last operation of a train is always chosen, so we can extract these cases
                    # In the latter case, iterating over the in_edges will suffice
                    if op_1 == 0 or op_1 == len(self.train_graphs[train_1].nodes) - 1:
                        self.model.add(op_1_chosen == 1)
                    else:
                        self.model.add(op_1_chosen == sum(self.edge_select_vars[train_1][in_edge]
                                                          for in_edge in self.train_graphs[train_1].in_edges(op_1)))

                    if op_2 == 0 or op_2 == len(self.train_graphs[train_2].nodes) - 1:
                        self.model.add(op_2_chosen == 1)
                    else:
                        self.model.add(op_2_chosen == sum(self.edge_select_vars[train_2][in_edge]
                                                          for in_edge in self.train_graphs[train_2].in_edges(op_2)))

                    # get the release time of the resource for both operations
                    rt_1 = self.find_release_time(train_1, op_1, res)
                    rt_2 = self.find_release_time(train_2, op_2, res)

                    end_1 = self.model.NewIntVar(lb=0, ub=2**20, name="Placeholder var")
                    end_2 = self.model.NewIntVar(lb=0, ub=2**20, name="Placeholder var")

                    self.model.add(end_1 == self.op_end_vars[train_1, op_1] + rt_1)
                    self.model.add(end_2 ==self.op_end_vars[train_2, op_2] + rt_2)

                    size_1 = self.model.new_int_var(lb=0, ub=2**20, name="Placeholder var")
                    size_2 = self.model.new_int_var(lb=0, ub=2**20, name="Placeholder var")


                    interval_1 = self.model.NewOptionalIntervalVar(start=self.op_start_vars[train_1, op_1],
                                                      end=end_1,
                                                      size=size_1,
                                                      is_present=op_1_chosen,
                                                      name=f"Interval for Train {train_1} : Operation {op_1}")

                    interval_2 = self.model.NewOptionalIntervalVar(start=self.op_start_vars[train_2, op_2],
                                                      end=end_2,
                                                      size=size_2,
                                                      is_present=op_2_chosen,
                                                      name=f"Interval for Train {train_2} : Operation {op_2}")

                    self.model.add_no_overlap([interval_1, interval_2])


    def add_deadlock_constraints(self):
        for cycle in tqdm(list(nx.simple_cycles(self.res_graph)), desc="Adding Deadlock-Constraints"):
            edges = []
            for edge in itertools.pairwise(cycle + [cycle[0]]):
                edge_data = self.res_graph[edge[0]][edge[1]].get("data", [])
                edges.append([(u,v,edge[0]) for u,v in edge_data])

            # sort edges by amount of trains
            edges.sort(key=lambda l: len(set([train for train, op, res in l])))
            
            # Do a quick pre-check to find cycles that would never create a deadlock
            if func_a(edges):
                continue

            self.func_b(edges, [], 0)


    def func_b(self, edges, current_tuple, depth):
        if depth == len(edges):
            self.model.add(sum(self.find_release_time(train, op, res) for train, op, res in current_tuple) >= 1)
            return

        for train_1, op_1, res_1 in edges[depth]:
            if all(train_1 != train_2 for train_2, op_2, res_2 in current_tuple):
                self.func_b(edges, current_tuple + [(train_1, op_1, res_1)], depth + 1)


    def set_objective(self):
        self.model.minimize(sum(obj["coeff"] * self.threshold_vars[obj["train"], obj["operation"]] + obj["increment"] * self.threshold_vars[obj["train"], obj["operation"]] for obj in self.objectives))


    def solve(self):
        if args.debug:
            self.save_res_graph_as_image()

        self.solver.parameters.log_search_progress = True

        status = self.solver.Solve(self.model)
        if status == cp.OPTIMAL or status == cp.FEASIBLE:
            if args.debug:
                self.save_train_graphs_as_image()
            print(f"Optimal Solution found with objective value found of {round(self.solver.objective_value)}")

            events = self.topological_sorted_events(self.get_events())
            return Solution(round(self.solver.objective_value), events)
        else:
            print(f"Model is infeasible!")
            return Solution(-1, [])


    def get_events(self):
        events = []
        for i, graph in enumerate(tqdm(self.train_graphs, desc="Creating events")):
            # This marks the start of the algorithm
            v = 0
            events.append({"time": self.solver.value(self.op_start_vars[(i, v)]), "train": i, "operation": v})

            # As long as the last operation wasn't reached yet...
            while v != len(graph.nodes) - 1:
                for succ in self.trains[i][v]["successors"]:
                    # ...find the chosen successor-operation...
                    if self.solver.value(self.edge_select_vars[i][(v, succ)]) == 1:
                        v = succ
                        # ...and add it to the events-list
                        events.append({"time": self.solver.value(self.op_start_vars[(i, v)]), "train": i, "operation": v})
                        break

        # Sort that list by "time" in a non-descending order. This will further help, since operations at the same time will be grouped together
        return sorted(events, key=lambda x: x["time"])


    def topological_sorted_events(self, events):
        sorted_events = []
        lhs, rhs = 0, 0

        while lhs < len(events):
            # Find the intervall of events that take place at the same time
            while rhs < len(events) and events[rhs]["time"] == events[lhs]["time"]:
                rhs += 1

            # If there are at least two events at the same time, we have to make sure they are sorted correctly
            if rhs - lhs > 1:
                topological_graph = self.create_topological_graph(events[lhs:rhs])

                # workaround cuz networkx is a bitch for not having implemented connected_components for digraphs (calm down pls)
                # Just found out about weakly_connected_components. Can't bother to change that for now (sigh)
                connected_graph = nx.Graph()
                connected_graph.add_nodes_from(topological_graph.nodes)
                connected_graph.add_edges_from(topological_graph.edges)

                for component in list(nx.connected_components(connected_graph)):
                    # Create that subgraph
                    subgraph = create_subgraph(topological_graph, component)

                    # Now, topologically sort it
                    for node in list(nx.topological_sort(subgraph)):
                        # reorder the group of events and add it to the sorted list
                        sorted_events.append(events[lhs + node])
            else:
                sorted_events.append(events[lhs])

            # update the starting index to find the next group of events
            lhs = rhs
        return sorted_events


    def create_topological_graph(self, events):
        # create a graph that represents the dependencies between operations: Graph may not be connected !!!
        graph = nx.DiGraph()
        graph.add_nodes_from(range(len(events)))

        # Add predecessor-edges
        for i, event in enumerate(events):
            # Retrieve the resources that need to be freed, so that the given operation can happen
            needed_resources = [res["resource"] for res in self.trains[event["train"]][event["operation"]]["resources"]]

            # Get the indices of events that need to take place before, so their predecessors resources are freed
            for j, other in enumerate(events):
                predecessor = self.get_predecessor(other["train"], other["operation"])

                # Of course an event is self-dependent. Skip that case to avoid self-loops. Otherwise, sorting will not work!
                if i == j or predecessor is None or event["train"] == other["train"]:
                    continue

                predecessor_resources = [res["resource"] for res in self.trains[other["train"]][predecessor]["resources"]]

                # If a predecessor of one of these events that took a needed resource, make sure the event takes place before the other event
                for res in needed_resources:
                    if res in predecessor_resources:
                        graph.add_edge(j, i)
                        break

        # Add priority-edges for same resources
        for i, event in enumerate(events):
            duration = self.solver.value(self.op_end_vars[(event["train"], event["operation"])]) - self.solver.value(self.op_end_vars[(event["train"], event["operation"])])
            critical_resources = []

            if duration > 0:
                # If the duration is non-zero, every operation that uses one of that resources needs to happen before it
                critical_resources.append(r["resource"] for r in self.trains[event["train"]][event["operation"]]["resources"])
            else:
                for r in self.trains[event["train"]][event["operation"]]["resources"]:
                    rt = 0
                    if type(r["release_time"]) != int:
                        rt = self.solver.value(r["release_time"])
                    else:
                        rt = r["release_time"]

                    # Else, only care about the resources with a non-zero release_time
                    if rt > 0:
                        critical_resources.append(r["resource"])

            for j, other in enumerate(events):
                if i == j or event["train"] == other["train"]:
                    continue

                # find other events j with same resource
                for other_r in self.trains[other["train"]][other["operation"]]["resources"]:
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
        for in_edge in self.train_graphs[train].in_edges(op):
            if self.solver.value(self.edge_select_vars[train][in_edge]) == 1:
                return in_edge[0]


    def find_release_time(self, train, operation, resource):
        for op_res in self.trains[train][operation]["resources"]:
            if op_res["resource"] == resource:
                return op_res["release_time"]

        return 0


    def save_train_graphs_as_image(self):
        for i, graph in enumerate(tqdm(self.train_graphs, desc="Creating Graphs")):
            colors = []

            for e in self.edge_select_vars[i].items():
                colors.append((200/255, 0, 0) if self.solver.value(e[1]) == 1.0 else (180/255, 180/255, 180/255))

            depth = nx.single_source_shortest_path_length(graph, 0)

            depth_groups = {}
            for node, d in depth.items():
                if d not in depth_groups:
                    depth_groups[d] = []
                depth_groups[d].append(node)

            pos = {}
            for d, nodes in depth_groups.items():
                count = len(nodes)
                for j, node in enumerate(sorted(nodes)):
                    pos[node] = (d, -((count - 1) / 2) + j)

            x = nx.shortest_path_length(graph, 0, len(graph.nodes) - 1)
            plt.figure(figsize=(x, 5))
            nx.draw(
                graph,
                pos,
                with_labels=True,
                node_size=500,
                font_size=10,
                arrowsize=10,
                node_color="lightgray",
                font_color="black",
                edge_color=colors,
                arrows=True
            )

            plt.savefig(f"Graphs/Train_{i}.png", format="png")
            plt.close()


    def save_res_graph_as_image(self):
        pos = nx.spring_layout(self.res_graph, seed=42)

        cycles = list(nx.simple_cycles(self.res_graph, 2))
        cycle_edges = [edge for cycle in cycles for edge in zip(cycle, cycle[1:] + [cycle[0]])]

        plt.figure(figsize=(40, 40))
        nx.draw(
            self.res_graph,
            pos,
            with_labels=True,
            node_size=500,
            font_size=10,
            arrowsize=10,
            node_color="lightgray",
            font_color="black",
            edge_color="lightgray",
            arrows=True
        )

        nx.draw_networkx_edges(
            self.res_graph,
            pos,
            edgelist=cycle_edges,
            edge_color="red",
            width=3.0
        )

        plt.savefig(f"Graphs/Resource_Graph.png", format="png")
        plt.close()


def main():
    try:
        with open(f"Testing/Instances/{args.instance}", 'r') as file:
            instance = json.load(file)
    except FileNotFoundError:
        print(f"File {args.instance} was not found")
        return

    instance = parse_instance(instance)
    solver = DisplibSolver(instance)
    solution = solver.solve()
    write_solution_to_file(solution)


def create_train_graph(train):
    graph = nx.DiGraph()
    graph.add_nodes_from([i for i, _ in enumerate(train)])

    for i, operation in enumerate(train):
        graph.add_edges_from([(i, v) for v in operation["successors"]])
    return graph


def create_subgraph(graph : nx.DiGraph, nodes):
    subgraph = nx.DiGraph()
    subgraph.add_nodes_from(nodes)
    subgraph.add_edges_from([edge for edge in graph.edges if edge[0] in nodes and edge[1] in nodes])
    return subgraph


def create_deadlock_graph(trains, resources):
    graph = nx.DiGraph()

    for i, train in enumerate(trains):
        for j, op in enumerate(train):
            for res in op["resources"]:
                # This is crucial for our idea: If the release time for this resource is greater than 0, the resource will be blocked although
                # the next operation has already started. This means that a deadlock would not be possible. Just don't draw the edge for that case
                if type(res["release_time"]) == int:
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


def func_a(edges):
    '''
    FIND MORE CRITERIA!!!
    '''
    unique_ids = set()
    for i, edge in enumerate(edges, 1):
        for train, op, res in edge:
            unique_ids.add(train)
        if len(unique_ids) < i:
            return True
    return False


def write_solution_to_file(solution : Solution):
    with open(f"Solutions/sol_{args.instance}", 'w') as file:
        file.write(json.dumps({"objective_value": solution.objective_value, "events": solution.events}))


def parse_instance(instance):
    # Fill in the defined default values for easy access
    for train in instance["trains"]:
        for operation in train:
            if "start_lb" not in operation:
                operation["start_lb"] = 0
            if "start_ub" not in operation:
                operation["start_ub"] = 2**20
            if "resources" not in operation:
                operation["resources"] = []
            else:
                for res_dict in operation["resources"]:
                    if "release_time" not in res_dict:
                        res_dict["release_time"] = 0

    for objective in instance["objective"]:
        if "threshold" not in objective:
            objective["threshold"] = 0
        if "increment" not in objective:
            objective["increment"] = 0
        if "coeff" not in objective:
            objective["coeff"] = 0

    return Instance(instance["trains"], instance["objective"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="DISPLIB-Solver", description="By Lina Breuer, Sebastian Brunke, Elias Kaiser, Felix Michel")
    parser.add_argument('instance', help="Filename of the instance that needs to be solved", type=str)
    parser.add_argument("--debug", action="store_true", help="Activates debug-mode")
    args = parser.parse_args()

    main()