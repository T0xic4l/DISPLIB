import argparse, json, os, time, logging, subprocess

from copy import deepcopy
from data import Instance
from heuristic import Heuristic
from logger import TimeLogger
from instance_properties import check_properties
from lns_coordinator import LnsCoordinator


path = os.path.join("Logs", "meine_logs.log")
os.makedirs(os.path.dirname(path), exist_ok=True)
file_handler = logging.FileHandler('log.txt', mode='w')
console_handler = logging.StreamHandler()
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s', handlers=[file_handler, console_handler])


def parse_instance(instance):
    # Fill in the defined default values for easy access
    for train in instance["trains"]:
        for op in train:
            if "start_lb" not in op:
                op["start_lb"] = 0
            if "start_ub" not in op:
                op["start_ub"] = 2 ** 40
            if "resources" not in op:
                op["resources"] = []
            else:
                op_res = set([res["resource"] for res in op["resources"]])
                resources = []
                for res in op_res:
                    for f_res in op["resources"]:
                        if f_res["resource"] == res:
                            if "release_time" not in f_res:
                                resources.append({"resource": res, "release_time": 1})
                            else:
                                resources.append({"resource": res, "release_time": f_res["release_time"]})
                            break
                op["resources"] = resources

    for k, objective in enumerate(instance["objective"]):
        if "threshold" not in objective:
            instance["objective"][k]["threshold"] = 0
        if "increment" not in objective:
            instance["objective"][k]["increment"] = 0
        if "coeff" not in objective:
            instance["objective"][k]["coeff"] = 0

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
        check_properties(instance, args.instance)
    else:
        solve_instance(instance)


def solve_instance(instance):
    res_eval, train_to_res = count_resource_appearances(instance.trains)
    log = Heuristic(deepcopy(instance), res_eval, train_to_res).schedule()

    if args.debug:
        log.write_final_solution_to_file("HeuristicSolutions", f"hsol_{args.instance}")
        log.write_log_to_file("Logs", f"log_{args.instance}")
        print(f"Found for {args.instance}. Elapsed time: {time.time() - log.start}")
    else:
        log.write_final_solution_to_file("Solutions", "temp.json")
        if not subprocess.run(f"python displib_verify.py Instances/{args.instance} Solutions/temp.json", shell=True, capture_output=True).returncode:
            LnsCoordinator(instance, log, res_eval, train_to_res, 600 - (time.time() - log.start)).solve()
            log.write_final_solution_to_file("Solutions", f"10min_sol2_{args.instance}")
        else:
            logging.critical("Heuristic-Solution is not valid!")


def count_resource_appearances(trains):
    resource_appearances = {}
    train_to_resources = {}

    for i, train in enumerate(trains):
        used_resources = set()
        for j, op in enumerate(train):
            for res in op["resources"]:
                used_resources.add(res["resource"])
        for res in used_resources:
            if resource_appearances.get(res) is not None:
                resource_appearances[res] += 1
            else:
                resource_appearances[res] = 1
        train_to_resources[i] = used_resources
    return resource_appearances, train_to_resources


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="DISPLIB-Solver", description="By Lina Breuer and Elias Kaiser")
    parser.add_argument('instance', help="Filename of the instance that needs to be solved. The solution of the instance will be saved in Solutions/ as sol_<instance>. The instance has to be located in Instances/", type=str)
    parser.add_argument('--checkproperties', action='store_true', help='Check for properties.')
    parser.add_argument('--debug', action='store_true', help='Activates debug-mode.')
    args = parser.parse_args()
    main()