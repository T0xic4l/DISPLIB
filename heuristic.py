# added verify to check heuristic solutions
# warning: There are some assumptions about the given instance
# 1. There is no release_time for any resource and for any operation
# 2. There is no start_ub for any operation except the first of each train

import copy, itertools
from ortools.sat.python import cp_model as cp


class FullInstanceHeuristic:
    def __init__(self, instance):
        self.instance = instance


    def full_instance_heuristic(self):
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


class HeuristicalSolver:
    def __init__(self, instance, time_limit):
        self.time_limit = time_limit
        self.instance = copy.deepcopy(instance)
        self.trains = instance.trains
        self.trains = self.choose_train_path()
        self.objectives = self.create_new_objective()
        self.model = cp.CpModel()
        self.solver = cp.CpSolver()
        self.trains_per_res = dict()  # A mapping from a resource to a list of operations using this operation
        self.op_start_vars = dict()
        self.op_end_vars = dict()
        self.threshold_vars = dict()

        for i, train in enumerate(self.trains):
            for j, op in enumerate(train):
                self.op_start_vars[i, j] = self.model.NewIntVar(lb=op["start_lb"], ub=op["start_ub"],
                                                                name=f"Start of Train {i} : Operation {j}")
                self.op_end_vars[i, j] = self.model.NewIntVar(lb=op["start_lb"] + op["min_duration"], ub=2 ** 20,
                                                              name=f"End of Train {i} : Operation {j}")

                # Create a mapping that maps a ressource to the list of operations using that ressource
                for res in op["resources"]:
                    if res["resource"] in self.trains_per_res.keys():
                        self.trains_per_res[res["resource"]].append((i, j))
                    else:
                        self.trains_per_res[res["resource"]] = [(i, j)]

        for obj in self.objectives:
            if obj["coeff"] != 0:
                self.threshold_vars[obj["train"], obj["operation"]] = self.model.NewIntVar(lb=0, ub=2 ** 20, name=f"Threshold of Train {obj["train"]} : Operation {obj["operation"]}")
            else:
                self.threshold_vars[obj["train"], obj["operation"]] = self.model.NewBoolVar(name=f"Threshold of Train {obj["train"]} : Operation {obj["operation"]}")

        self.add_threshold_constraints()
        self.add_timing_constraints()
        self.add_resource_constraints()
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
        sparse_trains = []
        for i, train in enumerate(self.trains):
            sparse_train = []
            op = 0
            operation = train[op]
            for res, used in enumerate(operation["resources"]):
                rel_time = used["release_time"]
                #rel_time = self.find_release_time(i, op, res)
                if rel_time == 0:
                    next_op = train[op]["successors"][0]
                    operation2 = train[next_op]
                    for res2, used2 in enumerate(operation2["resources"]):
                        if used2 == used:
                            continue
                    self.trains[i][op]["resources"][res]["release_time"] = 1
            self.trains[i][op]["operation"] = op
            sparse_train.append(self.trains[i][op])

            while op != len(train) - 1:
                op = train[op]["successors"][0]
                operation = train[op]
                for res, used_reso in enumerate(operation["resources"]):
                    rel_time = used_reso["release_time"]
                    if rel_time == 0:
                        next_op = train[op]["successors"][0]
                        operation2 = train[next_op]
                        for res2 in operation2["resources"]:
                            if res2 == res:
                                continue
                        self.trains[i][op]["resources"][res]["release_time"] = 1
                self.trains[i][op]["operation"] = op
                sparse_train.append(self.trains[i][op])
            sparse_trains.append(sparse_train)
        return sparse_trains


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
                    self.model.add_no_overlap([interval_vars[t1, op1], interval_vars[t2, op2]])


    def set_objective(self):
        self.model.minimize(sum(obj["coeff"] * self.threshold_vars[obj["train"], obj["operation"]] + obj["increment"] * self.threshold_vars[obj["train"], obj["operation"]] for obj in self.objectives))


    def solve(self):
        self.solver.parameters.log_search_progress = True
        self.solver.parameters.symmetry_level = 3
        self.solver.parameters.max_time_in_seconds = self.time_limit

        status = self.solver.Solve(self.model)

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
            return {-1: {"start": -1, "end": -1, "resources": []}}