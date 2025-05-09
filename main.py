import argparse, json, os, time, logging, subprocess, sys
import slurminade

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
                                resources.append({"resource": res, "release_time": 0})
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


def increase_release_times(instance):
    for train in instance.trains:
        for op in train:
            for res in op["resources"]:
                if not res["release_time"]:
                    res["release_time"] = 1
    return instance


@slurminade.slurmify()
def main(instance_path, time_limit, checkproperties, heuristicsol, debug):
    start = time.perf_counter()
    try:
        with open(os.path.join("Instances", instance_path), 'r') as file:
            instance = json.load(file)
    except FileNotFoundError:
        print(f"File {instance_path} was not found")
        return

    original_instance = parse_instance(instance)
    instance = increase_release_times(deepcopy(original_instance))

    if checkproperties:
        memory_handler = setup_logger()
        flush_logs_to_file(memory_handler, f"Logs/Properties/{instance_path}")
        check_properties(instance, instance_path)
    else:
        logging.info(f"Solving instance {instance_path}\n")
        memory_handler = setup_logger()
        sol = solve_instance(original_instance, instance, time_limit, (time.perf_counter()-start), heuristicsol, instance_path, debug)

        solution_written = write_solution_to_file(f"Solutions/10min_sol_{instance_path}", calculate_objective_value(instance.objectives, sol), sol)
        if solution_written:
            flush_logs_to_file(memory_handler, f"Logs/SolutionLogs/{os.path.splitext(instance_path)[0] + ".txt"}")
            if subprocess.run(f"python displib_verify.py Instances/{instance_path} Solutions/10min_sol_{instance_path}", shell=True, capture_output=True).returncode:
                logging.error("Final solution is not valid.")
            else:
                logging.info("Final solution is valid")


def solve_instance(original_instance, instance, time_limit, time_passed, heuristicsol, instance_path, debug):
    start = time.perf_counter()
    res_eval, train_to_res = count_resource_appearances(instance.trains)

    with TimeLogger("Calculating heuristic solution - "):
        h_result = Heuristic(deepcopy(instance), res_eval, train_to_res).schedule()

    if heuristicsol:
        write_solution_to_file(f"HeuristicSolutions/hsol_{instance_path}", calculate_objective_value(instance.objectives, h_result["solution"]), h_result["solution"], True)
        if debug:
            if subprocess.run(f"python displib_verify.py Instances/{instance_path} HeuristicSolutions/hsol_{instance_path}", shell=True, capture_output=True).returncode:
                logging.error("Heuristic solution is not valid.")
                sys.exit(1)
    return LnsCoordinator(original_instance, instance, h_result, res_eval, train_to_res, time_limit, time.perf_counter() - start).solve()


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


def write_solution_to_file(filename, objective, solution, force=False):
    if os.path.exists(filename) and not force:
        with open(filename, 'r') as file:
            try:
                prev_solution = json.load(file)
                if prev_solution["objective_value"] <= objective:
                    logging.info(f"A better or equal solution with obj_value {prev_solution["objective_value"]} for this instance already exists. Discarding current solution with obj_value {objective}.")
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
    # parser.add_argument('instance', help="Filename of the instance that needs to be solved. The solution of the instance will be saved in Solutions/ as sol_<instance>. The instance has to be located in Instances/", type=str)
    parser.add_argument('--timelimit', type=int, default=600, help="The time-limit in which the solution for the instance has to be calculated.")
    parser.add_argument('--checkproperties', action='store_true', help='Check for properties.')
    parser.add_argument('--heuristicsol', action='store_true', help='Print heuristic solution to file.')
    parser.add_argument('--debug', action='store_true', help='Activates debug-mode.')
    parser.add_argument('--slurm', action='store_true', help='Launches slurm jobs for each instance.')
    parser.add_argument('--pc', type=str, help='Chooses the pc. Available: pc01 | pc02 | pc03')
    args = parser.parse_args()
    slurminade.update_default_configuration(
        partition="alg",
        constraint="alggen03",
        exclusive=True,
        cpus_per_task=8,
        mem=32_000
    )

    if args.slurm:
        for instance_path in [f for f in os.listdir("Instances") if f.endswith(".json")]:
            main.distribute(instance_path, args.timelimit, args.checkproperties, args.heuristicsol, args.debug)
    else:
        for instance_path in [f for f in os.listdir(os.path.join("Instances", args.pc)) if f.endswith(".json")]:
            main(instance_path, args.timelimit, args.checkproperties, args.heuristicsol, args.debug)