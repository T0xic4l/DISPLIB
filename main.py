import argparse, json
import os

from data import Instance
from heuristic import get_heuristic_solution
from lns_coordinator import LnsCoordinator

def parse_instance(instance):
    # Fill in the defined default values for easy access
    for train in instance["trains"]:
        for operation in train:
            if "start_lb" not in operation:
                operation["start_lb"] = 0
            if "start_ub" not in operation:
                operation["start_ub"] = 2 ** 40
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

def main():
    try:
        with open(os.path.join("Instances", args.instance), 'r') as file:
            instance = json.load(file)
    except FileNotFoundError:
        print(f"File {args.instance} was not found")
        return

    instance = parse_instance(instance)
    heuristic_sol = get_heuristic_solution(instance)

    lns_coordinator = LnsCoordinator(instance, heuristic_sol, 600)
    log = lns_coordinator.log
    log.write_final_solution_to_file("Solutions", f"sol_{args.instance}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="DISPLIB-Solver", description="By Lina Breuer, Sebastian Brunke, Elias Kaiser, Felix Michel")
    parser.add_argument('instance', help="Filename of the instance that needs to be solved. The solution of the instance will be saved in Solutions/ as sol_<instance>. The instance has to be located in Instances/", type=str)
    parser.add_argument("--debug", action="store_true", help="Activates debug-mode. If set, a resource-allocation-graph and the operations_graph of each train (with chosen paths) will be created.")
    args = parser.parse_args()
    main()





