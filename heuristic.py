# added verify to check heuristic solutions
# warning: There are some assumptions about the given instance
# 1. There is no release_time for any resource and for any operation
# 2. There is no start_ub for any operation except the first of each train
import networkx as nx
import itertools
from tqdm import tqdm
from ortools.sat.python import cp_model as cp

from data import Solution
from logger import Log


class FirstSolver:
    def find_release_time(self, train, operation, resource):
        for op_res in self.trains[train][operation]["resources"]:
            if op_res["resource"] == resource:
                return op_res["release_time"]
        return 0

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
            #self.trains[i] = sparse_train
        return sparse_trains


    def __init__(self, instance):
        self.trains = instance.trains
        self.trains = self.choose_train_path()
        self.model = cp.CpModel()
        self.solver = cp.CpSolver()
        print(self.trains)
        self.trains_per_res = dict()  # A mapping from a resource to a list of operations using this operation
        self.op_start_vars = dict()
        self.op_end_vars = dict()

        for i, train in enumerate(self.trains):
            for j, op in enumerate(train):
                self.op_start_vars[i, j] = self.model.NewIntVar(lb=op["start_lb"], ub=op["start_ub"],
                                                                name=f"Start of Train {i} : Operation {j}")
                self.op_end_vars[i, j] = self.model.NewIntVar(lb=op["start_lb"] + op["min_duration"], ub=2 ** 20,
                                                              name=f"End of Train {i} : Operation {j}")

                # Create a mapping that maps a ressource to the list of operations using that ressource
                for res in op["resources"]:
                    #if res["release_time"] == 0:
                    #    res["release_time"] = self.model.NewBoolVar(name=f"rt for {(i, j)} : res {res}")

                    if res["resource"] in self.trains_per_res.keys():
                        self.trains_per_res[res["resource"]].append((i, j))
                    else:
                        self.trains_per_res[res["resource"]] = [(i, j)]

        #self.add_path_constraints()
        self.add_timing_constraints()
        self.add_resource_constraints()

    def add_timing_constraints(self):
        # Guarantee that every operation cannot start before the chosen predecessor (start + min_duration as a lower bound)
        for i, train in tqdm(enumerate(self.trains), desc="Adding timing-constraints"):
            self.model.add(self.op_start_vars[i, 0] == 0)
            for j, op in enumerate(train):

                    # If operation is chosen, successor may only start after at least start_var + min_duration
                self.model.add(self.op_start_vars[i, j] + self.trains[i][j]["min_duration"] <=
                               self.op_end_vars[i, j])
                    # Operation ends when successor operation starts
                if j == len(train)-1:
                    continue
                self.model.add(self.op_end_vars[i, j] == self.op_start_vars[i, j+1])

    def add_resource_constraints(self):
        for i, (res, ops) in enumerate(tqdm(self.trains_per_res.items(), desc="Adding resource-constraints")):
            # If there are multiple operations that use the same resource, a conflict could – in theory – be possible
            if len(ops) > 1:
                for (train_1, op_1), (train_2, op_2) in itertools.combinations(ops, 2):
                    # Since operations per train do not overlap due to the timing constraints, we can skip this case
                    if train_1 == train_2:
                        continue

                    # get the release time of the resource for both operations
                    rt_1 = self.find_release_time(train_1, op_1, res)
                    rt_2 = self.find_release_time(train_2, op_2, res)

                    end_1 = self.model.NewIntVar(lb=0, ub=2 ** 20, name="Placeholder var")
                    end_2 = self.model.NewIntVar(lb=0, ub=2 ** 20, name="Placeholder var")

                    self.model.add(end_1 == self.op_end_vars[train_1, op_1] + rt_1)
                    self.model.add(end_2 == self.op_end_vars[train_2, op_2] + rt_2)

                    size_1 = self.model.new_int_var(lb=0, ub=2 ** 20, name="Placeholder var")
                    size_2 = self.model.new_int_var(lb=0, ub=2 ** 20, name="Placeholder var")

                    interval_1 = self.model.NewIntervalVar(start=self.op_start_vars[train_1, op_1],
                                                                   end=end_1,
                                                                   size=size_1,
                                                                   name=f"Interval for Train {train_1} : Operation {op_1}")

                    interval_2 = self.model.NewIntervalVar(start=self.op_start_vars[train_2, op_2],
                                                                   end=end_2,
                                                                   size=size_2,
                                                                   name=f"Interval for Train {train_2} : Operation {op_2}")

                    self.model.add_no_overlap([interval_1, interval_2])

    def solve(self):
        self.solver.parameters.log_search_progress = True
        self.solver.parameters.max_time_in_seconds = 1200
        #self.solver.parameters.symmetry_level = 3

        status = self.solver.Solve(self.model)

        if status in (cp.FEASIBLE, cp.OPTIMAL):
            print(f"Initial solution found")
            solution = []
            verify = []
            for i, train in enumerate(self.trains):
                train_solution = {}
                verify_train = []
                for j, op in enumerate(train):
                    oper = op["operation"]
                    train_solution.update({oper: {"start": self.solver.value(self.op_start_vars[i,j]),
                                                  "end": self.solver.value(self.op_end_vars[i,j])}})
                    verify_train.append({"time": self.solver.value(self.op_start_vars[i,j]),
                                         "train": i,
                                         "operation": oper})
                solution.append(train_solution)
                verify.extend(verify_train)
            verify = sorted(verify, key=lambda x: x["time"])
            return solution, verify
        else:
            print(f"Model is infeasible!")
            return Log(status, -1, Solution(-1, []))

class Heuristic:
    def __init__(self,instance):
        self.current_time = 0
        self.instance = instance
        self.trains = instance.trains

    def get_heuristic_solution(self):
        solution = []
        verify = []
        for i, train in enumerate(self.trains):
            train_solution = {}
            verify_train = []
            op = 0
            end = max(train[0]["min_duration"], self.current_time)
            train_solution.update({0: {"start": 0, "end": end}})
            verify_train.append({"operation": 0,
                                 "time": 0,
                                 "train": i})

            self.current_time = train_solution[0]["end"]

            while op != len(train) - 1:
                op = train[op]["successors"][0]
                self.current_time = max(self.current_time, train[op]["start_lb"])
                end_time = self.current_time + train[op]["min_duration"]
                train_solution.update({op: {"start": self.current_time, "end": end_time}})

                verify_train.append({"operation": op,
                                     "time": self.current_time,
                                     "train": i})

                self.current_time = train_solution[op]["end"]

                verify.extend(verify_train)
            solution.append(train_solution)
        verify = sorted(verify, key=lambda x: x["time"])
        if self.conflict_search(verify):
            return False, False
        return solution, verify



    def conflict_search(self, events):
        current_res = {}
        train_res = {}
        for i, train in enumerate(self.trains):
            train_res[i] = None
        for event in events:
            train = event["train"]
            op = event["operation"]
            for res in self.trains[train][op]["resources"]:
                #resource is already in use
                if res["resource"] in current_res:
                    return True
                #resource is free:
                current_res[res["resource"]] = train
                old = train_res[train]
                if old:
                    current_res.pop(old)
                train_res[train] = res["resource"]
        return False

