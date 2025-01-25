import networkx as nx
import itertools
from tqdm import tqdm
from ortools.sat.python import cp_model as cp

from data import Solution, Instance
from logger import Log


class LnsDispblibSolver:
    def __init__(self, instance : Instance, feasible_solution, choice : list):
        self.trains = instance.trains
        self.objectives = instance.objectives
        self.train_graphs = instance.trains_graphs

        self.model = cp.CpModel()
        self.solver = cp.CpSolver()

        self.op_start_vars = dict()
        self.op_end_vars = dict()
        self.edge_select_vars = list()
        self.threshold_vars = dict()

        for i in choice:
            for j, op in enumerate(self.trains[i]):
                self.op_start_vars[i, j] = self.model.NewIntVar(lb=op["start_lb"], ub=op["start_ub"],
                                                                name=f"Start of Train {i} : Operation {j}")

                self.op_end_vars[i, j] = self.model.NewIntVar(lb=op["start_lb"] + op["min_duration"], ub=2 ** 20,
                                                              name=f"End of Train {i} : Operation {j}")

                for res in op["resources"]:
                    if res["release_time"] == 0:
                        res["release_time"] = self.model.NewBoolVar(name=f"rt for {(i, j)} : res {res}")

                if res["resource"] in self.trains_per_res.keys():
                    self.trains_per_res[res["resource"]].append((i, j))
                else:
                    self.trains_per_res[res["resource"]] = [(i, j)]

