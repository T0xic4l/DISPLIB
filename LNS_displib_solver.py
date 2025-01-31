import networkx as nx
import itertools
import copy
from ortools.sat.python import cp_model as cp

from data import Instance
from time import time
from tqdm import tqdm

from event_sorter import EventSorter


class LnsDisplibSolver:
    def __init__(self, instance : Instance, feasible_solution : list, choice : list, time_limit):
        self.time_limit = time_limit
        self.current_time = time()
        self.deadlock_constraints_added = True


        self.old_solution = copy.deepcopy(feasible_solution)
        self.feasible_sol = feasible_solution
        self.choice = choice

        self.instance = copy.deepcopy(instance)
        self.trains = self.instance.trains
        self.objectives = self.instance.objectives
        self.train_graphs = self.instance.get_train_graphs()

        self.model = cp.CpModel()
        self.solver = cp.CpSolver()

        self.resource_conflicts = dict()  # A mapping from a resource to a list of operations using this operation
        self.op_start_vars = dict()
        self.op_end_vars = dict()
        self.edge_select_vars = list()
        self.threshold_vars = dict()

        for i, train in enumerate(self.trains):
            if i in self.choice:
                select_vars = {}

                for j, op in enumerate(self.trains[i]):
                    self.op_start_vars[i, j] = self.model.NewIntVar(lb=op["start_lb"], ub=op["start_ub"],
                                                                    name=f"Start of Train {i} : Operation {j}")
                    self.op_end_vars[i, j] = self.model.NewIntVar(lb=op["start_lb"] + op["min_duration"], ub=2 ** 40,
                                                                  name=f"End of Train {i} : Operation {j}")

                    for res in op["resources"]:
                        if res["release_time"] == 0:
                            res["release_time"] = self.model.NewBoolVar(name=f"rt for {(i, j)} : res {res}")

                        if res["resource"] in self.resource_conflicts.keys():
                            self.resource_conflicts[res["resource"]].append((i, j))
                        else:
                            self.resource_conflicts[res["resource"]] = [(i, j)]

                    for s in op["successors"]:
                        select_vars[j, s] = self.model.NewBoolVar(name=f"Train {i} : Edge<{j},{s}>")
                self.edge_select_vars.append(select_vars)
            else:
                for op in self.feasible_sol[i].keys():
                    for res in self.trains[i][op]["resources"]:
                        if res["resource"] in self.resource_conflicts.keys():
                            self.resource_conflicts[res["resource"]].append((i, op))
                        else:
                            self.resource_conflicts[res["resource"]] = [(i, op)]

        for obj in self.objectives:
            if obj["train"] in self.choice:
                if obj["coeff"] != 0:
                    self.threshold_vars[obj["train"], obj["operation"]] = self.model.NewIntVar(lb=0, ub=2 ** 40, name=f"Threshold of Train {obj["train"]} : Operation {obj["operation"]}")
                else:
                    self.threshold_vars[obj["train"], obj["operation"]] = self.model.NewBoolVar(
                        name=f"Threshold of Train {obj["train"]} : Operation {obj["operation"]}")

        self.deadlock_graph = self.create_deadlock_graph()

        self.add_threshold_constraints()
        self.add_path_constraints()
        self.add_timing_constraints()
        self.add_resource_constraints()
        self.add_deadlock_constraints()
        self.set_objective()


    def set_objective(self):
        self.model.minimize(sum(obj["coeff"] * self.threshold_vars[obj["train"], obj["operation"]] + obj["increment"] * self.threshold_vars[obj["train"], obj["operation"]] for obj in self.objectives if obj["train"] in self.choice))


    def solve(self):
        if not self.deadlock_constraints_added:
            print("Too many cycles. Aborting!")
            return self.old_solution
        self.solver.parameters.log_search_progress = False
        self.solver.parameters.max_time_in_seconds = min(30, self.time_limit - time() + self.current_time) # This is just experimental to prevent time loss for expensive cycles
        status = self.solver.Solve(self.model)

        if status == cp.OPTIMAL or status == cp.FEASIBLE:
            '''
            At first, we thought that solving deadlocks by increasing release times could potentially destroy globally optimal solutions, since we alter the instance
            and release_times could not be reset once they are set.
            BUT it is possible that we get to optimize a certain train more than once. Because this train is variable again, we get the chance to reset an increased release-time
            IF another train solves the deadlock by increasing the release time, OR the cycle is not active because a the release time of a fixed train is 1. This is exactly what we want. 
            '''
            self.update_feasible_solution()
            return self.feasible_sol
        elif status == cp.UNKNOWN:
            print("Model is too large. Return the old solution")
            return self.old_solution
        else:
            print("Model is either infeasible or invalid. Return the old solution")
            return self.old_solution


    def add_threshold_constraints(self):
        for obj in self.objectives:
            train = obj["train"]

            if train in self.choice:
                op = obj["operation"]

                if obj["coeff"]:
                    self.model.add(self.threshold_vars[train, op] >= self.op_start_vars[train, op] - obj["threshold"])
                else:
                    self.model.add(self.op_start_vars[train, op] + 1 <= obj["threshold"]).OnlyEnforceIf(self.threshold_vars[train, op].Not())


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
                    else:
                        rt = 0
                        for f_res in self.feasible_sol[train][op]["resources"]:
                            if f_res["resource"] == res:
                                rt = f_res["release_time"]

                        interval_vars[train, op] = self.model.NewIntervalVar(start=self.feasible_sol[train][op]["start"],
                                                                                end=self.feasible_sol[train][op]["end"] + rt,
                                                                                size=self.feasible_sol[train][op]["end"] + rt - self.feasible_sol[train][op]["start"],
                                                                                name=f"Fix interval for Train {train} : Operation {op}")


                for (t1, op1), (t2, op2) in itertools.combinations(ops, 2):
                    if (t1 in self.choice or t2 in self.choice) and (t1 != t2):
                        self.model.add_no_overlap([interval_vars[t1, op1], interval_vars[t2, op2]])


    def add_deadlock_constraints(self):
        now = time()
        connected_comps = list(nx.weakly_connected_components(self.deadlock_graph))
        for comp in connected_comps:
            subgraph = EventSorter.create_subgraph(self.deadlock_graph, comp)

            for cycle in nx.simple_cycles(subgraph, len(self.trains)):
                if time() - now > 10:
                    self.deadlock_constraints_added = False
                    return


                all_train_edges = []  # Includes lists, one per edge
                var_train_edges = []  # Includes all (train, op, res) with train is in choice
                for edge in itertools.pairwise(cycle + [cycle[0]]):
                    all_edges = []
                    edge_data = self.deadlock_graph[edge[0]][edge[1]].get("data", [])

                    for train, op in edge_data:
                        all_edges.append((train, op, edge[0]))
                        if train in self.choice:
                            var_train_edges.append((train, op, edge[0]))
                    all_train_edges.append(all_edges)

                sum_vars = []
                for edges in all_train_edges:
                    sum_var = self.model.NewBoolVar(name="")
                    sum_vars.append(sum_var)
                    release_times = [self.find_release_time(train, op, res) for train, op, res in edges]
                    self.model.add(sum(release_times) == len(release_times)).OnlyEnforceIf(sum_var)

                all_rts_are_one = self.model.NewBoolVar(name="")
                sum_vars.append(all_rts_are_one)
                self.model.add(sum(self.find_release_time(train, op, res) for train, op, res in var_train_edges) == len(var_train_edges)).OnlyEnforceIf(all_rts_are_one)
                self.model.add(sum(sum_vars) >= 1)


    def update_feasible_solution(self):
        for i, train in enumerate(self.choice):
            train_sol = {}
            v = 0

            resource_list = []
            for res in self.trains[train][v]["resources"]:
                if type(res["release_time"]) == int:
                    resource_list.append(res)
                else:
                    resource_list.append({"resource": res["resource"], "release_time": round(self.solver.value(res["release_time"]))})

            train_sol.update({v: {"start": round(self.solver.value(self.op_start_vars[train, v])), "end": round(self.solver.value(self.op_end_vars[train, v])), "resources": resource_list}})

            while v != len(self.train_graphs[train].nodes) - 1:
                for succ in self.trains[train][v]["successors"]:
                    if round(self.solver.value(self.edge_select_vars[i][(v, succ)])) == 1:
                        v = succ

                        resource_list = []
                        for res in self.trains[train][v]["resources"]:
                            if type(res["release_time"]) == int:
                                resource_list.append(res)
                            else:
                                resource_list.append({"resource": res["resource"], "release_time": round(self.solver.value(res["release_time"]))})

                        train_sol.update({v: {"start": round(self.solver.value(self.op_start_vars[train, v])), "end": round(self.solver.value(self.op_end_vars[train, v])), "resources": resource_list}})
                        break

            self.feasible_sol[train] = train_sol


    def find_release_time(self, train, operation, resource):
        if train in self.choice:
            for op_res in self.trains[train][operation]["resources"]:
                if op_res["resource"] == resource:
                    return op_res["release_time"]
        else:
            for res in self.feasible_sol[train][operation]["resources"]:
                if res["resource"] == resource:
                    return res["release_time"]
        return 0


    def create_deadlock_graph(self):
        graph = nx.DiGraph()
        critical_resources = []
        for train in self.choice:
            for op in self.trains[train]:
                for res in op["resources"]:
                    critical_resources.append(res["resource"])


        for i, train in enumerate(self.trains):
            if i in self.choice:
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
            else:
                for op, value in self.feasible_sol[i].items():
                    for res in value["resources"]:
                        if res["release_time"] != 0:
                            continue

                        for succ in train[op]["successors"]:
                            for succ_res in train[succ]["resources"]:
                                edge = (res["resource"], succ_res["resource"])

                                if (edge[0] == edge[1]) or (edge[0] not in critical_resources) or (edge[1] not in critical_resources):
                                    continue
                                if edge in graph.edges:
                                    edge_data = graph[edge[0]][edge[1]].get("data", [])
                                    edge_data.append((i, op))
                                    graph[edge[0]][edge[1]]["data"] = edge_data
                                else:
                                    graph.add_nodes_from([res["resource"], succ_res["resource"]])
                                    graph.add_edge(edge[0], edge[1], data=[(i, op)])
        return graph