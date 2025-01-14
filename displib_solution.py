import argparse, json, itertools

import gurobipy as gp
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
Todos:
1. Done: objective
2. output
3. Konflikte finden
4. Model anhand konflikte aufteilen (max 3-4 Züge gleichzeitig)
5. mit Teilmodelen Konflikte beseitigen
6. Schritt 3-5 wiederholen bis Konflikt frei
"""

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
        self.model = gp.Model()

        self.trains_per_res = dict() # A mapping from a resource to a list of operations using this operation
        self.op_start_vars = dict()
        self.op_end_vars = dict()
        self.edge_select_vars = list()
        self.threshold_vars = dict()

        for i, train in enumerate(self.trains):
            select_vars = {}

            # inline declaration possible if ub is infinite per default
            for j, op in enumerate(train):
                self.op_start_vars[(i, j)] = self.model.addVar(vtype=gp.GRB.INTEGER, name=f"Start of Train {i} : Operation {j}")
                self.op_end_vars[(i, j)] = self.model.addVar(vtype=gp.GRB.INTEGER, name=f"End of Train {i} : Operation {j}", lb= op["start:lb"] + op["min_duration"])

                # Create a mapping that maps a ressource to the list of operations using that ressource
                for res in op["resources"]:
                    if res["resource"] in self.trains_per_res.keys():
                        self.trains_per_res[res["resource"]].append((i,j))
                    else:
                        self.trains_per_res[res["resource"]] = [(i,j)]

                for s in op["successors"]:
                    select_vars[(j, s)] = self.model.addVar(vtype=gp.GRB.BINARY,
                                                            name=f"Train {i} : Edge<{j},{s}>")
            self.edge_select_vars.append(select_vars)

            for obj in self.objectives:
                self.threshold_vars[(obj["train"], obj["operation"])] = self.model.addVar(vtype=gp.GRB.BINARY, name=f"Threshold of Train {obj["train"]} : Operation {obj["operation"]}")

        self.res_graph = create_res_graph(self.trains, self.trains_per_res.keys())

        # self.add_threshold_constraints()

        self.add_path_constraints()
        self.add_timing_constraints()
        self.add_resource_constraints()
        # self.add_deadlock_constraints()

        # self.set_objective()


    def add_threshold_constraints(self):
        for obj in self.objectives:
            train = obj["train"]
            op = obj["operation"]
            # If the threshold-var is false, we have to enforce that the start-time of the operation is below the threshold time
            self.model.addGenConstrIndicator(self.threshold_vars[(train, op)], False, self.op_start_vars[(train, op)] + 1, gp.GRB.LESS_EQUAL, obj["threshold"])


    def add_path_constraints(self):
        for i, train in tqdm(enumerate(self.trains), desc="Adding path-constraints"):
            # exactly one out_edge for first operation, one in_edge for last operation
            self.model.addConstr(gp.quicksum(self.edge_select_vars[i][out_edge]
                                             for out_edge in self.train_graphs[i].out_edges(0)) == 1)

            self.model.addConstr(gp.quicksum(self.edge_select_vars[i][in_edge]
                                             for in_edge in self.train_graphs[i].in_edges(len(self.train_graphs[i].nodes) - 1)) == 1)

            # If a vertex is not the first nor the last one, make sure the number of in edges is equal to the number of out edges
            # Not necessary to make sure there's max. 1 in_/out_edges selected, because first and last op already have only 1 edge selected
            for j in range(1, len(self.train_graphs[i].nodes) - 1):
                ins = self.train_graphs[i].in_edges(j)
                outs = self.train_graphs[i].out_edges(j)
                self.model.addLConstr(gp.quicksum(self.edge_select_vars[i][in_edge] for in_edge in ins) ==
                                     gp.quicksum(self.edge_select_vars[i][out_edge] for out_edge in outs))


    def add_timing_constraints(self):
        # Guarantee that every operation cannot start before the chosen predecessor (start + min_duration as a lower bound)
        for i, train in tqdm(enumerate(self.trains), desc="Adding timing-constraints"):
            # Since every train has to choose the first operation, enforce lower and upper bounds
            self.model.addLConstr(self.op_start_vars[(i, 0)] - self.trains[i][0]["start_lb"] >= 0)
            if self.trains[i][0]["start_ub"] is not None:
                self.model.addLConstr(self.op_start_vars[(i, 0)] - self.trains[i][0]["start_ub"] <= 0)

            for op in range(1, len(self.train_graphs[i].nodes)):
                for in_edge in self.train_graphs[i].in_edges(op):
                    # If operation is chosen, successor may only start after at least start_var + min_duration
                    self.model.addGenConstrIndicator(self.edge_select_vars[i][in_edge], True, self.op_start_vars[(i, in_edge[0])] + self.trains[i][in_edge[0]]["min_duration"] <= self.op_start_vars[(i, op)])

                    # Since this is not the first operation, iterating over in_edges will suffice.
                    self.model.addGenConstrIndicator(self.edge_select_vars[i][in_edge], True, self.op_start_vars[(i, op)] - self.trains[i][op]["start_lb"], gp.GRB.GREATER_EQUAL, 0)
                    if self.trains[i][op]["start_ub"] is not None:
                        self.model.addGenConstrIndicator(self.edge_select_vars[i][in_edge], True, self.op_start_vars[(i, op)] - self.trains[i][op]["start_ub"], gp.GRB.LESS_EQUAL, 0)

                for out_edge in self.train_graphs[i].out_edges(op):
                    # Operation ends when successor operation starts
                    self.model.addGenConstrIndicator(self.edge_select_vars[i][out_edge], True, self.op_end_vars[(i, op)] - self.op_start_vars[(i, out_edge[1])], gp.GRB.EQUAL, 0)

            '''
            # Since every train has to choose the last operation, enforce lower and upper bounds
            last_op = len(self.graphs[i].nodes) - 1
            self.model.addLConstr(self.op_start_vars[(i, last_op)] - self.trains[i][0]["start_lb"], gp.GRB.GREATER_EQUAL, 0)
            if self.trains[i][last_op]["start_ub"] is not None:
            self.model.addLConstr(self.op_start_vars[(i, last_op)] - self.trains[i][last_op]["start_ub"], gp.GRB.LESS_EQUAL, 0)
            '''


    def add_resource_constraints(self):
        # Note: A train may have conflicts with itself, so don't skip this case
        for i, (res, ops) in enumerate(tqdm(self.trains_per_res.items(), desc="Adding resource-constraints")):
            if len(ops) > 1:
                for (train_1, op_1), (train_2, op_2) in itertools.combinations(ops, 2):
                    if train_1 == train_2:
                        continue

                    # Boolvars for overlapping constraints
                    z1 = self.model.addVar(vtype=gp.GRB.BINARY)
                    z2 = self.model.addVar(vtype=gp.GRB.BINARY)

                    # search for the resource that this operation uses (multiple resources per operation possible)
                    rt_1 = self.find_release_time(train_1, op_1, res)
                    rt_2 = self.find_release_time(train_2, op_2, res)

                    # z1 => end1 + rt1 <= start2   ;   z2 => end2 + rt2 <= start1
                    self.model.addGenConstrIndicator(z1, True, self.op_end_vars[(train_1, op_1)] + rt_1 <= self.op_start_vars[(train_2, op_2)])
                    self.model.addGenConstrIndicator(z2, True, self.op_end_vars[(train_2, op_2)] + rt_2 <= self.op_start_vars[(train_1, op_1)])
                    # (Is logical equivalence required?! I guess not. The solver has to pick z1 or z2
                    # and therefore at least one inequality has to be satified)

                    # We solve resource-conflicts, if z1 or z2 is true: And we don't force trains to take a certain
                    # path, because start1, end1 is free, if op1 is not taken (same for z2), so the solver can always
                    # satisfy the constraint by just picking a start and an end
                    self.model.addConstr(z1 + z2 >= 1)


    def add_deadlock_constraints(self):
        # If there's a cycle of length 2, two operations are dependent from each other
        # How would this scale up for larger cycles?
        for cycle in tqdm(list(nx.simple_cycles(self.res_graph, 2)), desc="Adding Deadlock-Constraints"):
            if len(cycle) < 2:
                continue

            data1 = self.res_graph.edges[cycle[0], cycle[1]]["data"]
            data2 = self.res_graph.edges[cycle[1], cycle[0]]["data"]

            for (t1, (u, v)), (t2, (s, t)) in list(itertools.product(data1, data2)):
                # no deadlock in same train possible
                if t1 == t2:
                    continue

                # if not the same train, avoid deadlock by making u,v | s,t atomic
                z1 = self.model.addVar(vtype=gp.GRB.BINARY)
                z2 = self.model.addVar(vtype=gp.GRB.BINARY)

                self.model.addGenConstrIndicator(z1, True, self.op_end_vars[(t1, v)] <= self.op_start_vars[(t2, s)])
                self.model.addGenConstrIndicator(z2, True, self.op_end_vars[(t2, t)] <= self.op_start_vars[(t1, u)])

                # This enforces that the trains will not meet on this track. Either the first or second train goes through first
                self.model.addConstr(z1 + z2 >= 1)


    def find_release_time(self, train, operation, resource):
        for op_res in self.trains[train][operation]["resources"]:
            if op_res["resource"] == resource:
                return op_res["release_time"]

        return 0


    def solve(self):
        self.model.params.MIPGap = 0.50
        self.model.params.MIPFocus = 1
        self.model.optimize()

        if self.model.status == gp.GRB.OPTIMAL:
            if args.debug:
                self.save_train_graphs_as_image()
                self.save_res_graph_as_image()
            print(f"Optimal Solution found with objective value found of {self.model.objVal}")

            events = self.topological_sorted_events(self.get_events())
            return Solution(round(self.model.objVal), events)
        else:
            print(f"Model is infeasible!")
            return Solution(-1, [])


    def set_objective(self):
        # As always, gurobi has to ruin everything. Needed to add threshold vars so everything works
        self.model.setObjective(gp.quicksum(obj["coeff"] * self.threshold_vars[(obj["train"], obj["operation"])] * (self.op_start_vars[(obj["train"], obj["operation"])] - obj["threshold"]) +
                                             obj["increment"] * self.threshold_vars[(obj["train"], obj["operation"])]
                                            for obj in self.objectives), gp.GRB.MINIMIZE)


    def get_events(self):
        events = []
        for i, graph in enumerate(tqdm(self.train_graphs, desc="Creating events")):
            # This marks the start of the algorithm
            v = 0
            events.append({"time": round(self.op_start_vars[(i, v)].X), "train": i, "operation": v})

            # As long as the last operation wasn't reached yet...
            while v != len(graph.nodes) - 1:
                for succ in self.trains[i][v]["successors"]:
                    # ...find the chosen successor-operation...
                    if round(self.edge_select_vars[i][(v, succ)].X) == 1:
                        v = succ
                        # ...and add it to the events-list
                        events.append({"time": round(self.op_start_vars[(i, v)].X), "train": i, "operation": v})
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
            duration = round(self.op_end_vars[(event["train"], event["operation"])].X) - round(self.op_end_vars[(event["train"], event["operation"])].X)
            critical_resources = []

            if duration > 0:
                # If the duration is non-zero, every operation that uses one of that resources needs to happen before it
                critical_resources.append(r["resource"] for r in self.trains[event["train"]][event["operation"]]["resources"])
            else:
                # Else, only care about the resources with a non-zero release_time
                critical_resources.append(r["resource"] for r in self.trains[event["train"]][event["operation"]]["resources"] if r["release_time"] > 0)

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
            if round(self.edge_select_vars[train][in_edge].X) == 1:
                return in_edge[0]


    def save_train_graphs_as_image(self):
        for i, graph in enumerate(tqdm(self.train_graphs, desc="Creating Graphs")):
            colors = []

            for e in self.edge_select_vars[i].items():
                colors.append((200/255, 0, 0) if round(e[1].X) == 1.0 else (180/255, 180/255, 180/255))

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
        with open(f"Instances/{args.instance}", 'r') as file:
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


def create_res_graph(trains, resources):
    graph = nx.DiGraph()
    graph.add_nodes_from(resources)

    for i, train in enumerate(trains):
        for j, op in enumerate(train):
            for res in op["resources"]:
                for succ in op["successors"]:
                    for succ_res in train[succ]["resources"]:
                        edge = (res["resource"], succ_res["resource"])
                        if edge in graph.edges:
                            edge_data = graph[edge[0]][edge[1]].get("data", [])
                            edge_data.append((i, (j, succ)))
                            graph[edge[0]][edge[1]]["data"] = edge_data
                        else:
                            graph.add_edge(res["resource"], succ_res["resource"], data=[(i, (j, succ))])

    return graph


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
                operation["start_ub"] = None
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