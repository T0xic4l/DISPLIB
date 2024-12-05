import argparse
import json
import gurobipy as gp

from displib_verify import INFINITY


class Instance:
    def __init__(self, trains, objectives):
        self.trains = trains
        self.objectives = objectives


class DisplibSolver:
    def __init__(self, instance):
        self.trains = instance.trains
        self.objectives = instance.objectives
        self.model = gp.Model()

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
                # Maybe change that to INFINITY later...
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