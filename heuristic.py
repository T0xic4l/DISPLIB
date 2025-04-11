# added verify to check heuristic solutions
# warning: There are some assumptions about the given instance
# 1. There is no release_time for any resource and for any operation
# 2. There is no start_ub for any operation except the first of each train

import itertools, random
import time
from copy import deepcopy

import networkx as nx
from ortools.sat.python import cp_model as cp

from data import Instance


def calculate_heuristic_solution(instance : Instance):
    start = time.time()
    sequential_trains = get_sequential_trains(instance.trains)

    if sequential_trains:
        objective_a = [obj for obj in instance.objectives if obj["train"] in list(sequential_trains.keys())]
        sequential_sol = SequentialTrainScheduler(instance=Instance(list(sequential_trains.values()), objective_a)).solve()

        if len(sequential_trains) == len(instance.trains):
            return sequential_sol

        non_sequential_trains = {i: train for i, train in enumerate(instance.trains) if i not in list(sequential_trains.keys())}
        non_sequential_objectives = []
        train_mapping = dict()

        for i, train_nr in enumerate(non_sequential_trains.keys()):
            train_mapping[train_nr] = i
        for obj in instance.objectives:
            if obj["train"] in list(non_sequential_trains.keys()):
                # TODO: Check if copying is necessary
                obj_c = deepcopy(obj)
                obj_c["train"] = train_mapping[obj["train"]]
                non_sequential_objectives.append(obj_c)

        non_sequential_instance = Instance(list(non_sequential_trains.values()), non_sequential_objectives)

        # The heuristic will chose A SINGLE but RANDOM path through the operations graph for each train. Since it is possible, that this random choice leads to an unresolvable deadlock,
        # we have to restart the heuristic until a feasible solution is found
        while not (non_sequential_sol := TrainScheduler(non_sequential_instance, 600 - (time.time() - start)).schedule()):
            pass
        sol = merge_solutions(list(sequential_trains.keys()), sequential_sol, non_sequential_sol)
    else:
        while not (sol := TrainScheduler(instance, 600 - (time.time() - start)).schedule()):
            pass
    return sol


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


def get_sequential_trains(trains):
    sequential_trains = dict()

    for train_nr, train in enumerate(trains):
        if check_sequential_compatibility(train):
            sequential_trains[train_nr] = train

    return sequential_trains if len(sequential_trains) else None


def shift_operations(train, shift, fixed_op):
    for op, timings in train.items():
        if op > fixed_op:
            timings["start"] += shift
            timings["end"] += shift
        elif op == fixed_op:
            timings["end"] += shift


def check_sequential_compatibility(train):
    # check if first operation of the train does not use resources
    if train[0]["resources"]:
        return False

    # check if each operation except the first of a train has no default start_ub
    for _, op in enumerate(train, start=1):
        if op["start_ub"] > 2 ** 40:
            return False
    return True


class SequentialTrainScheduler:
    def __init__(self, instance):
        self.instance = instance

    def solve(self):
        solution = []
        current_time = 0
        max_pred_rt = 0

        for i, train in enumerate(self.instance.trains):

            train_solution = {}

            op = 0
            train_solution.update({0: {"start": 0, "end": max(train[0]["min_duration"], current_time + max_pred_rt), "resources": self.instance.trains[i][op]["resources"]}})
            current_time = train_solution[0]["end"]

            while op != len(train) - 1:
                op = train[op]["successors"][0]
                current_time = max(current_time, train[op]["start_lb"])
                train_solution.update({op: {"start": current_time, "end": current_time + train[op]["min_duration"], "resources": self.instance.trains[i][op]["resources"]}})
                current_time = train_solution[op]["end"]

            solution.append(train_solution)

            max_pred_rt = max(res["release_time"] for op in train for res in op["resources"])

        return solution


class FirstSolutionCallback(cp.CpSolverSolutionCallback):
    def __init__(self):
        super().__init__()


    def on_solution_callback(self):
        self.StopSearch()  # Suche abbrechen nach der ersten Lösung


class TrainScheduler:
    '''
    Was wir ab jetzt alles testen müssen:
    - Scheduled Züge nicht rausschmeißen sondern nach aktualisiertem Bewertungsschema neu planen. Eine konfliktfreie Route für alle existiert ja, aber vielleicht gibt es noch bessere
    - Ressourcen müssen sinnvoll bestraft werden. Selten aufkommende Ressourcen werden nicht automatisch direkt schlecht, wenn fast alle Vorkommen wirklich genutzt werden
    '''

    def __init__(self, instance, time_limit):
        self.instance = instance
        self.increment_release_times()
        self.trains = instance.trains
        self.objectives = instance.objectives
        self.resource_appearances = self.count_resource_appearances()


    def increment_release_times(self):
        for train in self.instance.trains:
            for op in train:
                for res in op["resources"]:
                    if res["release_time"] == 0:
                        res["release_time"] = 1


    def count_resource_appearances(self):
        resource_appearances = {}
        for train in self.trains:
            for i, op in enumerate(train):
                for res in op["resources"]:
                    if res["resource"] not in resource_appearances.keys():
                        resource_appearances[res["resource"]] = (1 - self.succ_uses_res(train, i, res["resource"]))
                    else:
                        resource_appearances[res["resource"]] += (1 - self.succ_uses_res(train, i, res["resource"]))

        return resource_appearances


    def succ_uses_res(self, train, op, res):
        for succ in train[op]["successors"]:
            for succ_res in train[succ]["resources"]:
                if succ_res["resource"] == res:
                    return 1
        return 0


    def schedule(self):
        feasible_solution = [{} for _ in range(len(self.trains))]                                                         # feasible solution format
        scheduled_trains = []
        unscheduled_trains = [i for i in range(len(self.trains))]

        while len(unscheduled_trains):
            to_schedule = random.choice(unscheduled_trains)
            sol = TrainSolver(self.instance, feasible_solution, scheduled_trains, to_schedule).solve()

            while not sol:
                reset_train = random.choice(scheduled_trains)
                print(f"No schedule for train {to_schedule}. Unscheduling {reset_train}")

                scheduled_trains.remove(reset_train)
                unscheduled_trains.append(reset_train)

                feasible_solution[reset_train] = {}
                sol = TrainSolver(self.instance, feasible_solution, scheduled_trains, to_schedule).solve()

            feasible_solution = sol
            unscheduled_trains.remove(to_schedule)
            scheduled_trains.append(to_schedule)
            scheduled_trains.sort()

        return feasible_solution


class TrainSolver:
    def __init__(self, instance, feasible_solution, fix_trains, choice):
        self.choice = [choice]
        self.fix_trains = fix_trains
        self.feasible_solution = feasible_solution
        self.trains = instance.trains
        self.train_graphs = self.create_train_graphs()

        self.model = cp.CpModel()
        self.solver = cp.CpSolver()
        self.callback = FirstSolutionCallback()

        self.resource_conflicts = dict()  # A mapping from a resource to a list of operations using this operation
        self.op_start_vars = dict()
        self.op_end_vars = dict()
        self.edge_select_vars = list()
        self.resource_usage_vars = dict()

        for i in self.choice:
            select_vars = {}
            for j, op in enumerate(self.trains[i]):
                self.op_start_vars[i, j] = self.model.NewIntVar(lb=op["start_lb"], ub=op["start_ub"],
                                                                name=f"Start of Train {i} : Operation {j}")
                self.op_end_vars[i, j] = self.model.NewIntVar(lb=op["start_lb"] + op["min_duration"], ub=2 ** 40,
                                                              name=f"End of Train {i} : Operation {j}")

                for res in op["resources"]:
                    if res["resource"] in self.resource_conflicts.keys():
                        self.resource_conflicts[res["resource"]].append((i, j))
                    else:
                        self.resource_conflicts[res["resource"]] = [(i, j)]

                for s in op["successors"]:
                    select_vars[j, s] = self.model.NewBoolVar(name=f"Train {i} : Edge<{j},{s}>")
            self.edge_select_vars.append(select_vars)

        for j in fix_trains:
            for op in self.feasible_solution[j].keys():
                for res in self.trains[j][op]["resources"]:
                    if res["resource"] in self.resource_conflicts.keys():
                        self.resource_conflicts[res["resource"]].append((j, op))
                    else:
                        self.resource_conflicts[res["resource"]] = [(j, op)]

        self.add_path_constraints()
        self.add_timing_constraints()
        self.add_resource_constraints()


    def add_path_constraints(self):
        for i, train in enumerate(self.choice):
            last_op = len(self.train_graphs[train].nodes) - 1

            self.model.add(sum(self.edge_select_vars[i][out_edge] for out_edge in self.train_graphs[train].out_edges(0)) == 1)
            self.model.add(sum(self.edge_select_vars[i][in_edge] for in_edge in self.train_graphs[train].in_edges(last_op)) == 1)

            for j in range(1, last_op):
                in_edges = self.train_graphs[train].in_edges(j)
                out_edges = self.train_graphs[train].out_edges(j)
                self.model.add(sum(self.edge_select_vars[i][in_edge] for in_edge in in_edges) ==
                               sum(self.edge_select_vars[i][out_edge] for out_edge in out_edges))


    def add_timing_constraints(self):
        for i, train in enumerate(self.choice):
            for op in range(len(self.train_graphs[train].nodes)):
                for in_edge in self.train_graphs[train].in_edges(op):
                    self.model.add(self.op_start_vars[train, in_edge[0]] + self.trains[train][in_edge[0]]["min_duration"] <= self.op_start_vars[train, op]).OnlyEnforceIf(self.edge_select_vars[i][in_edge])

                for out_edge in self.train_graphs[train].out_edges(op):
                    self.model.add(self.op_end_vars[train, op] == self.op_start_vars[train, out_edge[1]]).OnlyEnforceIf(self.edge_select_vars[i][out_edge])


    def add_resource_constraints(self):
        for res, ops in self.resource_conflicts.items():
            if len(ops) > 1:
                interval_vars = {}

                for train, op in ops:
                    if train in self.choice:
                        op_chosen = self.model.NewBoolVar(name=f"Train {train} : Operation {op} is chosen")
                        if op == 0 or op == len(self.train_graphs[train].nodes) - 1:
                            self.model.add(op_chosen == 1)
                        else:
                            for i, f_train in enumerate(self.choice):
                                if train == f_train:
                                    self.model.add(sum(self.edge_select_vars[i][in_edge] for in_edge in self.train_graphs[train].in_edges(op)) == op_chosen)

                        rt = self.find_release_time(train, op, res)
                        size = self.model.NewIntVar(lb=0, ub=2 ** 40, name=f"Placeholder var")
                        end = self.model.NewIntVar(lb=0, ub=2 ** 40, name=f"Placeholder var")
                        self.model.add(end == self.op_end_vars[train, op] + rt)

                        interval_vars[train, op] = self.model.NewOptionalIntervalVar(start=self.op_start_vars[train, op],
                                                                                        end=end,
                                                                                        size=size,
                                                                                        is_present=op_chosen,
                                                                                        name=f"Optional interval for Train {train} : Operation {op}")
                    elif train in self.fix_trains:
                        rt = 0
                        for f_res in self.feasible_solution[train][op]["resources"]:
                            if f_res["resource"] == res:
                                rt = f_res["release_time"]

                        interval_vars[train, op] = self.model.NewIntervalVar(start=self.feasible_solution[train][op]["start"],
                                                                                end=self.feasible_solution[train][op]["end"] + rt,
                                                                                size=self.feasible_solution[train][op]["end"] + rt - self.feasible_solution[train][op]["start"],
                                                                                name=f"Fix interval for Train {train} : Operation {op}")

                for (t1, op1), (t2, op2) in itertools.combinations(ops, 2):
                    if (t1 in self.choice or t2 in self.choice) and (t1 != t2):
                        self.model.add_no_overlap([interval_vars[t1, op1], interval_vars[t2, op2]])


    def solve(self):
        self.solver.parameters.log_search_progress = False
        status = self.solver.Solve(self.model)

        if status == cp.OPTIMAL or status == cp.FEASIBLE:
            self.update_feasible_solution()
            return self.feasible_solution
        else:
            return None


    def create_train_graphs(self):
        train_graphs = []
        for train in self.trains:
            graph = nx.DiGraph()
            graph.add_nodes_from([i for i, _ in enumerate(train)])

            for i, operation in enumerate(train):
                graph.add_edges_from([(i, v) for v in operation["successors"]])
            train_graphs.append(graph)
        return train_graphs


    def find_release_time(self, train, operation, resource):
        if train in self.choice:
            for op_res in self.trains[train][operation]["resources"]:
                if op_res["resource"] == resource:
                    return op_res["release_time"]
        else:
            for res in self.feasible_solution[train][operation]["resources"]:
                if res["resource"] == resource:
                    return res["release_time"]
        return 0


    def update_feasible_solution(self):
        for i, train in enumerate(self.choice):
            train_sol = {}
            v = 0

            resource_list = []
            for res in self.trains[train][v]["resources"]:
                resource_list.append(res)

            train_sol.update({v: {"start": round(self.solver.value(self.op_start_vars[train, v])), "end": round(self.solver.value(self.op_end_vars[train, v])), "resources": resource_list}})

            while v != len(self.train_graphs[train].nodes) - 1:
                for succ in self.trains[train][v]["successors"]:
                    if round(self.solver.value(self.edge_select_vars[i][(v, succ)])) == 1:
                        v = succ

                        resource_list = []
                        for res in self.trains[train][v]["resources"]:
                            resource_list.append(res)

                        train_sol.update({v: {"start": round(self.solver.value(self.op_start_vars[train, v])), "end": round(self.solver.value(self.op_end_vars[train, v])), "resources": resource_list}})
                        break

            self.feasible_solution[train] = train_sol