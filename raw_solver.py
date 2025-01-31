import itertools
from tqdm import tqdm
from ortools.sat.python import cp_model as cp


class RawSolver:
    def __init__(self, instance):
        self.trains = instance.trains
        self.objectives = instance.objectives
        self.train_graphs = instance.get_train_graphs()
        self.resource_conflicts = instance.get_resource_conflicts()  # A mapping from a resource to a list of operations using this operation
        self.deadlock_graph = instance.create_deadlock_graph()
        self.deadlocks = instance.get_deadlocks()

        self.model = cp.CpModel()
        self.solver = cp.CpSolver()

        self.op_start_vars = dict()
        self.op_end_vars = dict()
        self.edge_select_vars = list()
        self.threshold_vars = dict()

        for i, train in enumerate(self.trains):
            select_vars = {}

            for j, op in enumerate(train):
                self.op_start_vars[i, j] = self.model.NewIntVar(lb=op["start_lb"], ub=op["start_ub"],
                                                                name=f"Start of Train {i} : Operation {j}")
                self.op_end_vars[i, j] = self.model.NewIntVar(lb=op["start_lb"] + op["min_duration"], ub=2 ** 20,
                                                              name=f"End of Train {i} : Operation {j}")

                # If a release time is 0, a deadlock could in theory appear. For that scenario, create a boolvar so we can increase the release_time to 1
                for res in op["resources"]:
                    if not res["release_time"]:
                        res["release_time"] = self.model.NewBoolVar(name=f"rt for {(i, j)} : res {res}")

                for s in op["successors"]:
                    select_vars[j, s] = self.model.NewBoolVar(name=f"Train {i} : Edge<{j},{s}>")
            self.edge_select_vars.append(select_vars)

        for obj in self.objectives:
            if obj["coeff"] != 0:
                self.threshold_vars[obj["train"], obj["operation"]] = self.model.NewIntVar(lb=0, ub=2 ** 20, name=f"Threshold of Train {obj["train"]} : Operation {obj["operation"]}")
            else:
                self.threshold_vars[obj["train"], obj["operation"]] = self.model.NewBoolVar(name=f"Threshold of Train {obj["train"]} : Operation {obj["operation"]}")


        self.add_threshold_constraints()
        self.add_path_constraints()
        self.add_timing_constraints()
        self.add_resource_constraints()
        self.add_deadlock_constraints()

        self.set_objective()


    def solve(self):
        self.solver.parameters.log_search_progress = True
        self.solver.parameters.max_time_in_seconds = 600
        self.solver.parameters.symmetry_level = 3

        status = self.solver.Solve(self.model)

        if status == cp.OPTIMAL or status == cp.FEASIBLE:
            print(f"Optimal Solution found with objective value found of {round(self.solver.objective_value)}")
            feasible_sol = []

            for i, train in enumerate(self.trains):
                train_solution = {}
                for j, op in enumerate(train):
                    for succ in op["successors"]:
                        if self.solver.value(self.edge_select_vars[i][j, succ]):
                            train_solution.update({j: {"start": self.solver.value(self.op_start_vars[i, j]),
                                                        "end": self.solver.value(self.op_end_vars[i, j]),
                                                        "resources": op["resources"]}})
                    if j == len(train) - 1:
                        train_solution.update({j: {"start": self.solver.value(self.op_start_vars[i, j]),
                                                   "end": self.solver.value(self.op_end_vars[i, j]),
                                                   "resources": op["resources"]}})
                feasible_sol.append(train_solution)
            return feasible_sol
        else:
            print(f"Model is infeasible!")
            return {-1: {"start": -1, "end": -1, "resources": []}}


    def add_threshold_constraints(self):
        for i, obj in tqdm(enumerate(self.objectives), desc="Adding threshold-constraints"):
            train = obj["train"]
            op = obj["operation"]

            # There is no delay if start = threshold in case coeff != 0
            if obj["coeff"] != 0:
                self.model.add(
                    self.threshold_vars[obj["train"], obj["operation"]] >= self.op_start_vars[train, op] - obj[
                        "threshold"])
            else:
                # But there is a delay if start = threshold if increment != 0 (check out the objective-formular in 2.1.1)
                self.model.add(self.op_start_vars[train, op] + 1 <= obj["threshold"]).OnlyEnforceIf(
                    self.threshold_vars[train, op].Not())


    def add_path_constraints(self):
        for i, train in tqdm(enumerate(self.trains), desc="Adding path-constraints"):
            last_op = len(self.train_graphs[i].nodes) - 1

            # For the first operation, exactly one outgoing edge must be chosen
            # For the last operation, exactly one ingoing edge must be chosen
            self.model.add(sum(self.edge_select_vars[i][out_edge] for out_edge in self.train_graphs[i].out_edges(0)) == 1)
            self.model.add(sum(self.edge_select_vars[i][in_edge] for in_edge in self.train_graphs[i].in_edges(last_op)) == 1)

            # For all operations inbetween, the number of chosen outgoing and ingoing edges must be equal
            for j in range(1, last_op):
                in_edges = self.train_graphs[i].in_edges(j)
                out_edges = self.train_graphs[i].out_edges(j)
                self.model.add(sum(self.edge_select_vars[i][in_edge] for in_edge in in_edges) == sum(self.edge_select_vars[i][out_edge] for out_edge in out_edges))


    def add_timing_constraints(self):
        # Guarantee that every operation cannot start before the chosen predecessor (start + min_duration as a lower bound)
        for i, train in tqdm(enumerate(self.trains), desc="Adding timing-constraints"):
            for op in range(len(self.train_graphs[i].nodes)):
                for in_edge in self.train_graphs[i].in_edges(op):
                    # If operation is chosen, successor may only start after at least start_var + min_duration
                    self.model.add(self.op_start_vars[i, in_edge[0]] + self.trains[i][in_edge[0]]["min_duration"] <= self.op_start_vars[i, op]).OnlyEnforceIf(self.edge_select_vars[i][in_edge])

                for out_edge in self.train_graphs[i].out_edges(op):
                    # Operation ends when successor operation starts
                    self.model.add(self.op_end_vars[i, op] == self.op_start_vars[i, out_edge[1]]).OnlyEnforceIf(self.edge_select_vars[i][out_edge])


    def add_resource_constraints(self):
        for res, ops in tqdm(self.resource_conflicts.items(), desc="Adding resource-constraints"):
            # If there are multiple operations that use the same resource, a conflict could – in theory – be possible
            if len(ops) > 1:
                interval_vars = {}

                for train, op in ops:
                    op_chosen = self.model.NewBoolVar(name=f"Train {train} : Operation {op} is chosen")
                    if op == 0 or op == len(self.train_graphs[train].nodes) - 1:
                        self.model.add(op_chosen == 1)
                    else:
                        self.model.add(sum(self.edge_select_vars[train][in_edge] for in_edge in self.train_graphs[train].in_edges(op)) == op_chosen)

                    rt = self.find_release_time(train, op, res)
                    size = self.model.NewIntVar(lb=0, ub=2 ** 40, name=f"Placeholder var")
                    end = self.model.NewIntVar(lb=0, ub=2 ** 40, name=f"Placeholder var")
                    self.model.add(end == self.op_end_vars[train, op] + rt)

                    interval_vars[train, op] = self.model.NewOptionalIntervalVar(start=self.op_start_vars[train, op],
                                                                                 end=end,
                                                                                 size=size,
                                                                                 is_present=op_chosen,
                                                                                 name=f"Optional interval for Train {train} : Operation {op}")

                for (t1, op1), (t2, op2) in itertools.combinations(ops, 2):
                    if t1 == t2:
                        continue
                    else:
                        self.model.add_no_overlap([interval_vars[t1, op1], interval_vars[t2, op2]])


    def add_deadlock_constraints(self):
        for cycle in tqdm(self.deadlocks, desc="Adding Deadlock-Constraints"):
            edges = []
            for edge in itertools.pairwise(cycle + [cycle[0]]):
                edge_data = self.deadlock_graph[edge[0]][edge[1]].get("data", [])
                edges.append([(u, v, edge[0]) for u, v in edge_data])

            # sort edges by amount of trains
            edges.sort(key=lambda l: len(set([train for train, op, res in l])))

            # Do a quick pre-check to find cycles that would never create a deadlock
            if not check_n_trains_in_n_cycle(edges):
                continue

            self.find_cycle_tuple(edges, [], 0)


    def set_objective(self):
        self.model.minimize(sum(
            obj["coeff"] * self.threshold_vars[obj["train"], obj["operation"]] + obj["increment"] * self.threshold_vars[
                obj["train"], obj["operation"]] for obj in self.objectives))



    def find_cycle_tuple(self, edges, current_tuple, depth):
        if depth == len(edges):
            self.model.add(sum(self.find_release_time(train, op, res) for train, op, res in current_tuple) >= 1)
            return

        for train_1, op_1, res_1 in edges[depth]:
            if all(train_1 != train_2 for train_2, op_2, res_2 in current_tuple):
                self.find_cycle_tuple(edges, current_tuple + [(train_1, op_1, res_1)], depth + 1)


    def find_release_time(self, train, operation, resource):
        for op_res in self.trains[train][operation]["resources"]:
            if op_res["resource"] == resource:
                return op_res["release_time"]
        return 0


def check_n_trains_in_n_cycle(edges):
    unique_ids = set()
    for i, edge in enumerate(edges, 1):
        for train, op, res in edge:
            unique_ids.add(train)
        if len(unique_ids) < i:
            return False
    return True