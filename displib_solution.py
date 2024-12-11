import argparse
import json
import gurobipy as gp
import networkx as nx
import matplotlib.pyplot as plt


class Instance:
    def __init__(self, trains, objectives):
        self.trains = trains
        self.objectives = objectives


class DisplibSolver:
    def __init__(self, instance):
        self.trains = instance.trains
        self.objectives = instance.objectives
        self.graphs = [create_graph(train) for train in self.trains]

        if args.debug:
            save_graphs_as_image(self.graphs)

        self.model = gp.Model()
        self.op_start_vars = []
        self.edge_select_vars = []

        for i, train in enumerate(self.trains):
            start_vars = []
            select_vars = {}

            # inline declaration possible if ub is infinite per default
            for j, op in enumerate(train):
                if op["start_ub"]:
                    start_vars.append(self.model.addVar(vtype=gp.GRB.INTEGER, lb=op["start_lb"], ub=op["start_ub"],
                                                        name=f"Train {i} : Operation {j}"))
                else:
                    start_vars.append(self.model.addVar(vtype=gp.GRB.INTEGER, lb=op["start_lb"],
                                                        name=f"Train {i} : Operation {j}"))

                for s in op["successors"]:
                    select_vars[(j, s)] = self.model.addVar(vtype=gp.GRB.BINARY,
                                                            name=f"Train {i} : Edge<{j},{s}>")
            self.op_start_vars.append(start_vars)
            self.edge_select_vars.append(select_vars)

        self.add_path_constrains()
        self.add_timing_constraints()


    def add_path_constrains(self):
        for i, train in enumerate(self.trains):
            # exactly one out_edge for first operation, one in_edge for last operation
            self.model.addConstr(gp.quicksum(self.edge_select_vars[i][out_edge]
                                             for out_edge in self.graphs[i].out_edges(0)) == 1)

            self.model.addConstr(gp.quicksum(self.edge_select_vars[i][in_edge]
                                             for in_edge in self.graphs[i].in_edges(len(self.graphs[i].nodes) - 1)) == 1)

            # If a vertex is not the first nor the last one, make sure the number of in egdes is equal to the number of out edges
            # Not necessary to make sure there's max. 1 in_/out_edges selected, because first and last op already have only 1 edge selected
            for j in range(1, len(self.graphs[i].nodes) - 1):
                ins = self.graphs[i].in_edges(j)
                outs = self.graphs[i].out_edges(j)
                self.model.addConstr(gp.quicksum(self.edge_select_vars[i][in_edge] for in_edge in ins) ==
                                     gp.quicksum(self.edge_select_vars[i][out_edge] for out_edge in outs))

    def add_timing_constraints(self):
        return

    def solve(self):
        self.model.optimize()

        if self.model.status == gp.GRB.OPTIMAL:
            for v in self.model.getVars():
                print(f"{v.VarName} = {v.X}")

        # Just to check for now
        events = [  {"time": 0, "train": 0, "operation": 0},
                    {"time": 0, "train": 1, "operation": 0},
                    {"time": 5, "train": 0, "operation": 2},
                    {"time": 5, "train": 1, "operation": 1},
                    {"time": 10, "train": 1, "operation": 2},
                    {"time": 10, "train": 0, "operation": 3}]

        return Solution(10, events)

    def set_objective(self):
        # self.model.setObjective(gp.quicksum(for ob in self.objectives))
        return 0

class Solution:
    def __init__(self, objective_value, events):
        self.objective_value = objective_value
        self.events = events


def main():
    try:
        with open(f"Instances/{args.instance}", 'r') as file:
            instance = json.load(file)
    except FileNotFoundError:
        print(f"File {args.instance} was not found")
        return
    except json.JSONDecodeError:
        print(f"File {args.instance} could not be decoded")
        return

    instance = parse_instance(instance)

    solver = DisplibSolver(instance)
    solution = solver.solve()
    write_solution_to_file(solution)


def create_graph(train):
    graph = nx.DiGraph()
    graph.add_nodes_from([i for i, _ in enumerate(train)])

    for i, operation in enumerate(train):
        graph.add_edges_from([(i, v) for v in operation["successors"]])
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


def save_graphs_as_image(graphs):
    print("Creating graphs...")

    for i, graph in enumerate(graphs):
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
            node_size=700,
            node_color="lightblue",
            font_size=10,
            font_color="black",
            edge_color="gray",
            arrows=True,
            arrowsize=20
        )

        plt.savefig(f"Graphs/graph{i}.png", format="png")
        plt.close()
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="DISPLIB-Solver", description="By Lina Breuer, Sebastian Brunke, Elias Kaiser, Felix Michel")
    parser.add_argument('instance', help="Filename of the instance that needs to be solved", type=str)
    parser.add_argument("--debug", action="store_true", help="Activates debug-mode")
    args = parser.parse_args()

    main()