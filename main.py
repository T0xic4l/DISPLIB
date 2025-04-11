import argparse, json, os
import time
from copy import deepcopy

from data import Instance
from heuristic import calculate_heuristic_solution
from logger import Log
from instance_properties import check_properties
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

    if args.checkproperties:
        check_properties(instance)
    else:
        solve_instance(instance)


def solve_instance(instance):
    start = time.time()
    sol = calculate_heuristic_solution(deepcopy(instance))

    if args.debug:
        log = Log(sol, instance.objectives)
        log.write_final_solution_to_file("HeuristicSolutions", f"heuristic_sol_{args.instance}")
        print(f"Found for {args.instance}. Elapsed time: {time.time() - start}")
    else:
        coordinator = LnsCoordinator(instance, sol, 600 - (time.time() - start))
        coordinator.solve()
        coordinator.log.write_final_solution_to_file("Solutions", f"10min_sol2_{args.instance}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="DISPLIB-Solver", description="By Lina Breuer and Elias Kaiser")
    parser.add_argument('instance', help="Filename of the instance that needs to be solved. The solution of the instance will be saved in Solutions/ as sol_<instance>. The instance has to be located in Instances/", type=str)
    parser.add_argument('--checkproperties', action='store_true', help='Check for properties.')
    parser.add_argument('--debug', action='store_true', help='Activates debug-mode.')
    args = parser.parse_args()
    main()