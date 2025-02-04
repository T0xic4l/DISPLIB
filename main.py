import argparse, json, os, math, copy
import time
from copy import deepcopy

import lns_coordinator
from data import Instance
from heuristic import FullInstanceHeuristic, HeuristicalSolver, FirstSolutionCallback
from raw_solver import RawSolver
from logger import Log
from lns_coordinator import LnsCoordinator
import networkx as nx


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
        with open(os.path.join("Phase1Instances", args.instance), 'r') as file:
            instance = json.load(file)
    except FileNotFoundError:
        print(f"File {args.instance} was not found")
        return

    start = time.time()
    instance = parse_instance(instance)

    if args.rawsolve:
        log = RawSolver(instance).solve()
        log.write_final_solution_to_file("Solutions", f"raw_sol_{args.instance}")
        log.write_log_to_file("Logs", f"raw_log_{args.instance}")
        return

    while True:
        sol = calculate_heuristic_solution(instance)
        if sol is not None:
            break

    coordinator = LnsCoordinator(instance, sol, 600 - (time.time() - start))
    coordinator.solve()
    coordinator.log.write_final_solution_to_file("Solutions", f"10min_sol_{args.instance}")


def calculate_heuristic_solution(instance : Instance):
    start = time.time()
    partial_instance_a = split_for_heuristic_compatibility(instance.trains)

    if partial_instance_a:
        objective_a = [obj for obj in instance.objectives if obj["train"] in list(partial_instance_a.keys())]
        sol_a = FullInstanceHeuristic(instance=Instance(list(partial_instance_a.values()), objective_a)).full_instance_heuristic()

        if len(partial_instance_a) == len(instance.trains):
            return sol_a

        partial_instance_b = {i: train for i, train in enumerate(instance.trains) if i not in list(partial_instance_a.keys())}
        list_b = list(partial_instance_b.values())
        objective_b = []
        train_mapping = dict()

        for i, train_nr in enumerate(partial_instance_b.keys()):
            train_mapping[train_nr] = i
        for obj in instance.objectives:
            if obj["train"] in list(partial_instance_b.keys()):
                obj_c = deepcopy(obj)
                obj_c["train"] = train_mapping[obj["train"]]
                objective_b.append(obj_c)


        sol_b = HeuristicalSolver(Instance(list_b, objective_b), 30).solve()
        sol = merge_solutions(list(partial_instance_a.keys()), sol_a, sol_b)
        return sol
    else:
        return HeuristicalSolver(instance, 600 - (time.time() - start)).solve()

def merge_solutions(trains_a, sol_a, sol_b,):
    '''
    Params: sol_a has to be the sol of the compatible part
    compatible_trains is a list of train_nrs that were compatible: sol_a is the sol of them
    sol_b trains will start first because they block resources at their first op
    '''
    train_count = (len(sol_a) + len(sol_b))
    sol = [None] * train_count
    end = 0
    max_rt = 0

    for train in sol_b:
        for op, timings in train.items():
            for res in timings["resources"]:
                max_rt = max(max_rt, res["release_time"])
            end = max(end, timings["end"])

    for i, train in enumerate(trains_a):
        shift_operations(sol_a[i], end + max_rt, 0)
        sol[train] = sol_a[i]

    none_counter = 0
    for i, s in enumerate(sol):
        if not s:
            sol[i] = sol_b[none_counter]
            none_counter += 1

    pass
    return sol


def split_for_heuristic_compatibility(trains):
    compatible_trains = dict()
    for train_nr, train in enumerate(trains):
        if check_heuristic_compatibility([train]):
            compatible_trains[train_nr] = train
    comp = len(compatible_trains)
    total = len(trains)
    pass
    return compatible_trains if len(compatible_trains) else None


def shift_operations(train, shift, fixed_op):
    for op, timings in train.items():
        if op > fixed_op:
            timings["start"] += shift
            timings["end"] += shift
        elif op == fixed_op:
            timings["end"] += shift


def decide_solving_strategy(instance : Instance):
    '''
    Decides Solving Strategy

    0: Take the SolverHeuristic and LNS
    1: Take the FullInstance-Heuristic and LNS
    2: Take the displib_solver as one
    '''
    heuristic_compatibility = check_heuristic_compatibility(instance.trains)

    '''
    resource_conflicts = create_resource_conflict_mapping(instance).items()
    deadlock_graph = create_deadlock_graph(instance)
    # cycles = list(nx.simple_cycles(deadlock_graph))

    # ~~~ absolut statistics ~~~
    trains = len(instance.trains)
    operations = sum(len(train) for train in instance.trains)
    choice_nodes = sum(1 for train in instance.trains for op in train if len(op["successors"]) > 1)
    resources = len(resource_conflicts)
    resource_conflicts_in_total = sum(len(ops) for res, ops in resource_conflicts)
    # deadlocks = len(cycles)

    # ~~~ relative statistics ~~~
    average_resource_conflicts_per_resource = resource_conflicts_in_total/resources
    # average_cycle_length = sum(len(cycle) for cycle in cycles) / deadlocks
    '''

    if heuristic_compatibility:
        return 1
    else:
        return 0


def create_deadlock_graph(instance : Instance):
    graph = nx.DiGraph()
    for i, train in enumerate(instance.trains):
        for j, op in enumerate(train):
            for res in op["resources"]:
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


def create_resource_conflict_mapping(instance : Instance):
    resource_conflicts = dict()
    for t, train in enumerate(instance.trains):
        # Create a mapping that maps a ressource to the list of operations using that ressource
        for o, op in enumerate(train):
            for res in op["resources"]:

                if res["resource"] in resource_conflicts.keys():
                    resource_conflicts[res["resource"]].append((t, o))
                else:
                    resource_conflicts[res["resource"]] = [(t, o)]
    return resource_conflicts


def check_heuristic_compatibility(trains):
    # check if every first operation does not use resources
    for train in trains:
        if len(train[0]["resources"]):
            return False

    # check if each operation except the 0. of a train has no default start_ub
    for train in trains:
        for _, op in enumerate(train, start=1):
            if op["start_ub"] > 2 ** 40:
                return False

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="DISPLIB-Solver", description="By Lina Breuer, Sebastian Brunke, Elias Kaiser, Felix Michel")
    parser.add_argument('instance', help="Filename of the instance that needs to be solved. The solution of the instance will be saved in Solutions/ as sol_<instance>. The instance has to be located in Instances/", type=str)
    parser.add_argument("--rawsolve", action="store_true", help="If set, the instance will be solved a whole.")
    args = parser.parse_args()
    main()