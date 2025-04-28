import argparse, json, os, time, logging, subprocess, sys

from copy import deepcopy
from data import Instance
from heuristic import Heuristic
from event_sorter import EventSorter
from logger import TimeLogger
from instance_properties import check_properties
from lns_coordinator import LnsCoordinator
from logging.handlers import MemoryHandler


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
    start = time.perf_counter()
    time_limit = args.timelimit
    try:
        with open(os.path.join("Instances", args.instance), 'r') as file:
            instance = json.load(file)
    except FileNotFoundError:
        print(f"File {args.instance} was not found")
        return

    instance = parse_instance(instance)

    if args.checkproperties:
        memory_handler = setup_logger()
        flush_logs_to_file(memory_handler, f"Logs/Properties/{args.instance}")
        check_properties(instance, args.instance)
    else:
        memory_handler = setup_logger()
        sol = solve_instance(instance, time_limit, (time.perf_counter()-start))

        solution_written = write_solution_to_file(f"Solutions/10min_sol_{args.instance}", calculate_objective_value(instance.objectives, sol), sol)
        if solution_written:
            flush_logs_to_file(memory_handler, f"Logs/SolutionLogs/{os.path.splitext(args.instance)[0] + ".txt"}")
            if subprocess.run(f"python displib_verify.py Instances/{args.instance} Solutions/10min_sol_{args.instance}", shell=True, capture_output=True).returncode:
                logging.error("Final solution is not valid.")
            else:
                logging.info("Final solution is valid")


def solve_instance(instance, time_limit, time_passed):
    start = time.perf_counter()
    res_eval, train_to_res = count_resource_appearances(instance.trains)

    with TimeLogger("Calculating heuristic solution - "):
        h_result = Heuristic(deepcopy(instance), res_eval, train_to_res).schedule()

    if args.heuristicsol:
        write_solution_to_file(f"HeuristicSolutions/hsol_{args.instance}", calculate_objective_value(instance.objectives, h_result["solution"]), h_result["solution"])
        if args.debug:
            if subprocess.run(f"python displib_verify.py Instances/{args.instance} HeuristicSolutions/hsol_{args.instance}", shell=True, capture_output=True).returncode:
                logging.error("Heuristic solution is not valid.")
                sys.exit(1)
    return LnsCoordinator(instance, h_result, res_eval, train_to_res, time_limit, time.perf_counter() - start).solve()


def setup_logger():
    memory_handler = MemoryHandler(
        capacity=10000,
        flushLevel=logging.CRITICAL + 1,
        target=None)

    console_handler = logging.StreamHandler()
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s',
        handlers=[memory_handler, console_handler]
    )
    return memory_handler


def flush_logs_to_file(memory_handler, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_handler = logging.FileHandler(path, mode='w')
    memory_handler.setTarget(file_handler)
    memory_handler.flush()
    file_handler.close()


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


def write_solution_to_file(filename, objective, solution):
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            try:
                prev_solution = json.load(file)
                if prev_solution["objective_value"] <= objective:
                    logging.info("A better or equal solution for this instance already exists. Discarding current solution.")
                    return False
            except json.JSONDecodeError:
                logging.warning("Existing solution could not be parsed. Overwriting.")

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as file:
        json.dump({"objective_value": objective, "events": EventSorter(deepcopy(solution)).events}, file)
        return True

def calculate_objective_value(objectives, solution):
    return sum(obj["coeff"] * max(0, solution[obj["train"]][obj["operation"]]["start"] - obj["threshold"])
               + obj["increment"] * int(solution[obj["train"]][obj["operation"]]["start"] >= obj["threshold"])
               for obj in objectives if obj["operation"] in solution[obj["train"]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="DISPLIB-Solver", description="By Lina Breuer and Elias Kaiser")
    parser.add_argument('instance', help="Filename of the instance that needs to be solved. The solution of the instance will be saved in Solutions/ as sol_<instance>. The instance has to be located in Instances/", type=str)
    parser.add_argument('--timelimit', type=int, default=600, help="The time-limit in which the solution for the instance has to be calculated.")
    parser.add_argument('--checkproperties', action='store_true', help='Check for properties.')
    parser.add_argument('--debug', action='store_true', help='Activates debug-mode.')
    parser.add_argument('--heuristicsol', action='store_true', help='Print heuristic solution to file.')
    args = parser.parse_args()
    main()

