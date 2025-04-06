# added verify to check heuristic solutions
# warning: There are some assumptions about the given instance
# 1. There is no release_time for any resource and for any operation
# 2. There is no start_ub for any operation except the first of each train

import copy, itertools, random
import time
from copy import deepcopy
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
        while not (non_sequential_sol := NonSequentialTrainScheduler(non_sequential_instance, 30).solve()):
            pass
        sol = merge_solutions(list(sequential_trains.keys()), sequential_sol, non_sequential_sol)
    else:
        while not (sol := NonSequentialTrainScheduler(instance, 600 - (time.time() - start)).solve()):
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


class NonSequentialTrainScheduler:
    def __init__(self, instance, time_limit):
        self.time_limit = time_limit

        self.instance = copy.deepcopy(instance)
        self.trains = self.choose_train_path()
        self.objectives = self.create_new_objective()

        self.model = cp.CpModel()
        self.solver = cp.CpSolver()

        self.trains_per_res = dict()  # A mapping from a resource to a list of operations using this operation
        self.op_start_vars = dict()
        self.op_end_vars = dict()
        self.threshold_vars = dict()

        self.callback = FirstSolutionCallback()

        for i, train in enumerate(self.trains):
            for j, op in enumerate(train):
                self.op_start_vars[i, j] = self.model.NewIntVar(lb=op["start_lb"], ub=op["start_ub"],
                                                                name=f"Start of Train {i} : Operation {j}")
                self.op_end_vars[i, j] = self.model.NewIntVar(lb=op["start_lb"] + op["min_duration"], ub=2 ** 40,
                                                              name=f"End of Train {i} : Operation {j}")

                # Create a mapping that maps a ressource to the list of operations using that ressource
                for res in op["resources"]:
                    if j != 0 and not res["release_time"]:
                        res["release_time"] = 1
                    if res["resource"] in self.trains_per_res.keys():
                        self.trains_per_res[res["resource"]].append((i, j))
                    else:
                        self.trains_per_res[res["resource"]] = [(i, j)]

        for obj in self.objectives:
            if obj["coeff"] != 0:
                self.threshold_vars[obj["train"], obj["operation"]] = self.model.NewIntVar(lb=0, ub=2 ** 40, name=f"Threshold of Train {obj["train"]} : Operation {obj["operation"]}")
            else:
                self.threshold_vars[obj["train"], obj["operation"]] = self.model.NewBoolVar(name=f"Threshold of Train {obj["train"]} : Operation {obj["operation"]}")

        self.add_threshold_constraints()
        self.add_timing_constraints()
        self.add_strict_resource_constraints()
        self.set_objective()


    def find_release_time(self, train, operation, resource):
        for op_res in self.trains[train][operation]["resources"]:
            if op_res["resource"] == resource:
                return op_res["release_time"]
        return 0


    def create_new_objective(self):
        trains_and_ops = [[op["operation"] for op in train] for train in self.trains]
        new_objective = []

        for obj in self.instance.objectives:
            if obj["operation"] in trains_and_ops[obj["train"]]:
                new_objective.append(obj)

        return new_objective


    def choose_train_path(self):
        new_trains = []
        for i, train in enumerate(self.instance.trains):
            op_nr = 0
            op = train[op_nr]
            op["operation"]= op_nr

            single_path_train = [op]

            while op_nr != len(train) - 1:
                op_nr = random.sample(train[op_nr]["successors"], 1)[0]
                op = train[op_nr]
                op["operation"] = op_nr
                single_path_train.append(op)

            new_trains.append(single_path_train)

        return new_trains


    def add_threshold_constraints(self):
        for obj in self.objectives:
            train = obj["train"]
            op = obj["operation"]

            op_index = -1
            for i, oper in enumerate(self.trains[train]):
                if oper["operation"] == op:
                    op_index = i

            if obj["coeff"]:
                self.model.add(self.threshold_vars[train, op] >= self.op_start_vars[train, op_index] - obj["threshold"])
            else:
                self.model.add(self.op_start_vars[train, op_index] + 1 <= obj["threshold"]).OnlyEnforceIf(self.threshold_vars[train, op].Not())


    def add_timing_constraints(self):
        for i, train in enumerate(self.trains):
            self.model.add(self.op_start_vars[i, 0] == 0)

            for j, op in enumerate(train):
                if j != len(train) - 1:
                    self.model.add(self.op_end_vars[i, j] == self.op_start_vars[i, j + 1])
                    self.model.add(self.op_start_vars[i, j] + self.trains[i][j]["min_duration"] <= self.op_end_vars[i, j])
                else:
                    self.model.add(self.op_start_vars[i, j] + self.trains[i][j]["min_duration"] == self.op_end_vars[i, j])


    def add_strict_resource_constraints(self):
        interval_vars = dict() # one per operation
        for i, train in enumerate(self.trains):
            for j, op in enumerate(train):
                max_rt = 0
                for res in op["resources"]:
                    max_rt = max(max_rt, res["release_time"])
                interval_size = self.model.NewIntVar(lb=0, ub=2 ** 20, name=f"Placeholder var")
                interval_vars[i, j] = self.model.NewIntervalVar(start=self.op_start_vars[i, j],
                                                                     end=self.op_end_vars[i, j] + max_rt,
                                                                     size=interval_size,
                                                                     name=f"Fix interval for Train {i} : Operation {j}")

        for res, ops in self.trains_per_res.items():
            if len(ops) > 1:
                for (t1, op1), (t2, op2) in itertools.combinations(ops, 2):
                    if t1 != t2:
                        self.model.add_no_overlap([interval_vars[t1, op1], interval_vars[t2, op2]])


    def add_resource_constraints(self):
        for res, ops in self.trains_per_res.items():
            if len(ops) > 1:
                interval_vars = {}

                for train, op in ops:
                    release_time = self.find_release_time(train, op, res)
                    interval_size = self.model.NewIntVar(lb=0, ub=2 ** 20, name=f"Placeholder var")

                    interval_vars[train, op] = self.model.NewIntervalVar(start=self.op_start_vars[train, op],
                                              end=self.op_end_vars[train, op] + release_time,
                                              size=interval_size,
                                              name=f"Fix interval for Train {train} : Operation {op}")

                for (t1, op1), (t2, op2) in itertools.combinations(ops, 2):
                    if t1 != t2:
                        self.model.add_no_overlap([interval_vars[t1, op1], interval_vars[t2, op2]])


    def set_objective(self):
        self.model.minimize(sum(obj["coeff"] * self.threshold_vars[obj["train"], obj["operation"]] + obj["increment"] * self.threshold_vars[obj["train"], obj["operation"]] for obj in self.objectives))


    def solve(self):
        self.solver.parameters.log_search_progress = True
        self.solver.parameters.symmetry_level = 3
        self.solver.parameters.max_time_in_seconds = self.time_limit

        status = self.solver.SolveWithSolutionCallback(self.model, self.callback)
        # status = self.solver.Solve(self.model)

        if status in (cp.FEASIBLE, cp.OPTIMAL):
            print(f"Initial solution found")
            solution = []

            for i, train in enumerate(self.trains):
                train_solution = {}
                for j, op in enumerate(train):
                    oper = op["operation"]

                    train_solution.update({oper: {"start": self.solver.value(self.op_start_vars[i, j]),
                                                "end": self.solver.value(self.op_end_vars[i, j]),
                                                "resources": op["resources"]}})
                solution.append(train_solution)
            return solution
        else:
            print(f"Model is infeasible!")
            return None