import argparse, json, os, copy
from data import Instance
from heuristic import FullInstanceHeuristic, HeuristicalSolver
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
        with open(os.path.join("Instances", args.instance), 'r') as file:
            instance = json.load(file)
    except FileNotFoundError:
        print(f"File {args.instance} was not found")
        return

    instance = parse_instance(instance)
    solving_strategy = 0
    match solving_strategy:
        case 0:
            heuristic_sol = HeuristicalSolver(instance).solve()
            # heuristic_sol = FullInstanceHeuristic(instance).full_instance_heuristic()
        case _:
            heuristic_sol = None

    if heuristic_sol is not None:
        lns_coordinator = LnsCoordinator(instance, heuristic_sol, 540)
        log = lns_coordinator.log
        log.write_final_solution_to_file("Solutions", f"sol_{args.instance}")


def decide_solving_strategy(instance : Instance):
    '''
    Decides Solving Strategy
    :return:
    0: Take the displib_solver as one
    1: Take the FullInstance-Heuristic and LNS
    2: Take the SolverHeuristic and LNS
    '''
    heuristic_compatibility = check_heuristic_compatibility(instance)
    trains = len(instance.trains)
    operations = sum([len(train) for train in instance.trains])

    choice_nodes = 0

    for train in instance.trains:
        for i, op in enumerate(train):
            if len(op["successors"]) > 1:
                choice_nodes += 1

    resource_conflicts_in_total = sum(len(ops) for res, ops in create_resource_conflict_mapping(instance).items())
    average_resource_conflicts_per_resource = resource_conflicts_in_total/trains

    deadlock_graph = create_deadlock_graph(instance)
    deadlocks = list(nx.simple_cycles(deadlock_graph))
    average_cycle_length = sum(len(cycle) for cycle in deadlocks)


def create_deadlock_graph(instance : Instance):
        graph = nx.DiGraph()
        for i, train in enumerate(instance.trains):
            for j, op in enumerate(train):
                for res in op["resources"]:

                    if type(res["release_time"]) == int:
                        continue

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


def check_heuristic_compatibility(instance : Instance):
    # check if every first operation does not use resources
    for train in instance.trains:
        if len(train[0]["resources"]):
            return False

    # check if each operation except the 0. of a train has no default start_ub
    for train in instance.trains:
        for _, op in enumerate(train, start=1):
            if op["start_ub"] > 2 ** 40:
                return False
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="DISPLIB-Solver", description="By Lina Breuer, Sebastian Brunke, Elias Kaiser, Felix Michel")
    parser.add_argument('instance', help="Filename of the instance that needs to be solved. The solution of the instance will be saved in Solutions/ as sol_<instance>. The instance has to be located in Instances/", type=str)
    parser.add_argument("--debug", action="store_true", help="Activates debug-mode. If set, a resource-allocation-graph and the operations_graph of each train (with chosen paths) will be created.")
    args = parser.parse_args()
    main()





