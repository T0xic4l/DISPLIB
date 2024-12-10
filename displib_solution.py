import argparse
import json
import gurobipy as gp
import networkx as nx


class Instance:
    def __init__(self, trains, objectives):
        self.trains = trains
        self.objectives = objectives


class DisplibSolver:
    def __init__(self, instance):
        self.trains = instance.trains
        self.objectives = instance.objectives
        self.graphs = [create_graph(train) for train in self.trains]

        self.model = gp.Model()
        self.start_vars = []
        self.select_vars = []

        for i, train in enumerate(self.trains):
            op_start_vars = []
            op_select_vars = []

            # inline declaration possible if ub is infinite per default
            for j, op in enumerate(train):
                if op["start_ub"]:
                    op_start_vars.append(self.model.addVar(vtype=gp.GRB.INTEGER, lb=op["start_lb"], ub=op["start_ub"],
                                                           name=f"Start Train {i} : Operation {j}"))
                else:
                    op_start_vars.append(self.model.addVar(vtype=gp.GRB.INTEGER, lb=op["start_lb"]))
                op_select_vars.append(self.model.addVar(vtype=gp.GRB.BINARY, name=f"Select Train {i} : Operation {j}"))
            self.start_vars.append(op_start_vars)
            self.select_vars.append(op_select_vars)


    def solve(self):
        self.model.optimize()

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="DISPLIB-Solver", description="By Lina Breuer, Sebastian Brunke, Elias Kaiser, Felix Michel")
    parser.add_argument('instance', help="Filename of the instance that needs to be solved", type=str)
    args = parser.parse_args()

    main()