import networkx as nx
import itertools
import copy
import logging

from networkx.algorithms.cycles import simple_cycles
from ortools.sat.python import cp_model as cp

from data import Instance
from time import time
from collections import defaultdict
from copy import deepcopy

from event_sorter import EventSorter
from heuristic import check_resource_avoidance
from logger import TimeLogger


class LnsDisplibSolver:
    def __init__(self, instance : Instance, feasible_solution, choice, semi_fixed, train_to_resources, time_limit, deadlocks=False):
        logging.debug(f"Variable Trains: {choice}")
        self.time_limit = time_limit
        self.current_time = time()
        self.deadlocks = deadlocks
        self.deadlock_constraints_added = True

        self.old_solution = copy.deepcopy(feasible_solution)
        self.feasible_sol = copy.deepcopy(feasible_solution)
        self.choice = choice
        self.semi_fixed = semi_fixed
        self.fix_trains = [i for i in range(len(instance.trains)) if i not in self.choice + self.semi_fixed]

        self.instance = copy.deepcopy(instance)
        self.trains = self.instance.trains
        self.objectives = self.instance.objectives
        self.train_graphs = self.create_train_graphs()
        self.train_to_resources = train_to_resources

        self.model = cp.CpModel()
        self.solver = cp.CpSolver()

        self.op_start_vars = dict()
        self.op_end_vars = dict()
        self.edge_select_vars = defaultdict(dict)
        self.threshold_vars = dict()

        with TimeLogger("Model Setup"):
            self.create_objective_vars()
            self.resource_conflicts = defaultdict(lambda : defaultdict(lambda :defaultdict(list)))
            self.resource_usage_vars = defaultdict(dict)
            self.create_vars()

        with TimeLogger("Constraints"):
            self.add_threshold_constraints()
            self.add_path_constraints()
            self.add_timing_constraints()
            self.add_resource_constraints()
            self.add_resource_usage_constraints()
            if self.deadlocks:
                self.deadlock_graph = self.create_deadlock_graph()
                self.add_deadlock_constraints()
            self.add_solution_hint()

        self.set_objective()


    def solve(self):
        self.solver.parameters.log_search_progress = False
        self.solver.parameters.num_workers = 8

        if not self.deadlock_constraints_added:
            logging.warning("Too many deadlock constraints. Returning the old solution")
            return self.old_solution

        if time() - self.current_time >= self.time_limit:
            logging.info("No time to solve the model. Returning the old solution")
            return self.old_solution

        self.solver.parameters.max_time_in_seconds = max(0, self.time_limit - time() + self.current_time) # This is just experimental to prevent time loss for expensive cycles
        status = self.solver.Solve(self.model)

        if status == cp.OPTIMAL or status == cp.FEASIBLE:
            self.update_feasible_solution()
            return self.feasible_sol
        elif status == cp.INFEASIBLE:
            logging.warning("Model is infeasible. Returning the old solution.")
            return self.old_solution
        elif status == cp.MODEL_INVALID:
            logging.warning("Model is invalid. Returning the old solution.")
            return self.old_solution
        else:
            logging.warning("Model is unknown. Returning the old solution.")
            return self.old_solution


    def set_objective(self):
        self.model.minimize(sum(obj["coeff"] * self.threshold_vars[obj["train"], obj["operation"]] + obj["increment"] * self.threshold_vars[obj["train"], obj["operation"]]
                                for obj in self.objectives if obj["train"] in self.choice or (obj["train"] in self.semi_fixed and obj["operation"] in list(self.feasible_sol[obj["train"]].keys()))))


    def create_vars(self):
        with TimeLogger("Creating vars"):
            for i in self.choice:
                res_to_ops = defaultdict(list)
                for j, op in enumerate(self.trains[i]):
                    self.op_start_vars[i, j] = self.model.NewIntVar(lb=op["start_lb"], ub=op["start_ub"], name="")
                    self.op_end_vars[i, j] = self.model.NewIntVar(lb=op["start_lb"] + op["min_duration"], ub=2 ** 40, name="")

                    for res in op["resources"]:
                        res_to_ops[res["resource"]].append(j)

                        if self.deadlocks:
                            if not res["release_time"]:
                                res["release_time"] = self.model.NewBoolVar(name="")

                    for s in op["successors"]:
                        self.edge_select_vars[i][j, s] = self.model.NewBoolVar(name="")

                for res in self.train_to_resources[i]:
                    if check_resource_avoidance(self.trains[i], [res], return_path=False):
                        self.resource_usage_vars[i][res] = self.model.NewBoolVar(name="")


                for res in self.train_to_resources[i]:
                    start_ops = res_to_ops[res].copy()
                    end_ops = res_to_ops[res].copy()

                    for op in res_to_ops[res]:
                        if all(any(succ_res["resource"] == res for succ_res in self.trains[i][succ]["resources"]) for succ in self.trains[i][op]["successors"]):
                            end_ops.remove(op)

                        if op != 0:
                            predecessors = [pred[0] for pred in self.train_graphs[i].in_edges(op)]
                            if all(any(pred_res["resource"] == res for pred_res in self.trains[i][pred]["resources"]) for pred in predecessors):
                                start_ops.remove(op)

                    self.resource_conflicts[res][i]["start"] = start_ops
                    self.resource_conflicts[res][i]["end"] = end_ops

            for i in self.semi_fixed:
                for op, timings in self.feasible_sol[i].items():
                    self.op_start_vars[i, op] = self.model.NewIntVar(lb=self.trains[i][op]["start_lb"], ub=self.trains[i][op]["start_ub"], name="")
                    self.op_end_vars[i, op] = self.model.NewIntVar(lb=self.trains[i][op]["start_lb"] + self.trains[i][op]["min_duration"], ub=2 ** 40, name="")

                    for res in timings["resources"]:
                        if not self.resource_conflicts.get(res["resource"]):
                            self.resource_conflicts[res["resource"]][i]["start"] = []
                            self.resource_conflicts[res["resource"]][i]["end"] = []

            for i in self.fix_trains + self.semi_fixed:
                pred = -1
                track_resources = list()
                for op, timings in self.feasible_sol[i].items():
                    op_res = [res["resource"] for res in timings["resources"]]
                    for res in op_res:
                        if res not in track_resources and self.resource_conflicts.get(res) is not None:
                            track_resources.append(res)
                            self.resource_conflicts[res][i]["start"].append(op)
                    if pred != -1:
                        track_copy = deepcopy(track_resources)
                        for res in track_copy:
                            if res not in op_res:
                                track_resources.remove(res)
                                self.resource_conflicts[res][i]["end"].append(pred)
                    pred = op


    def create_objective_vars(self):
        for obj in self.objectives:
            if obj["train"] in self.choice or (obj["train"] in self.semi_fixed and obj["operation"] in list(self.feasible_sol[obj["train"]].keys())):
                if obj["coeff"] != 0:
                    self.threshold_vars[obj["train"], obj["operation"]] = self.model.NewIntVar(lb=0, ub=2 ** 40, name="")
                else:
                    self.threshold_vars[obj["train"], obj["operation"]] = self.model.NewBoolVar(name="")


    def add_solution_hint(self):
        for train in self.choice:
            pre = None
            for op, timings in self.feasible_sol[train].items():
                if pre is not None:
                    self.model.add_hint(self.edge_select_vars[train][(pre, op)], 1)
                pre = op
                self.model.add_hint(self.op_start_vars[train, op], timings["start"])
                self.model.add_hint(self.op_end_vars[train, op], timings["end"])

        for train in self.semi_fixed:
            for op, timings in self.feasible_sol[train].items():
                self.model.add_hint(self.op_start_vars[train, op], timings["start"])
                self.model.add_hint(self.op_end_vars[train, op], timings["end"])


    def add_threshold_constraints(self):
        with TimeLogger("Threshold Constraints"):
            for obj in self.objectives:
                train = obj["train"]

                if train in self.choice:
                    op = obj["operation"]
                    op_selected = self.model.new_bool_var(name="")
                    if op != 0:
                        self.model.add(op_selected == sum(self.edge_select_vars[train][in_e] for in_e in self.train_graphs[train].in_edges(op)))
                    else:
                        self.model.add(op_selected == sum(self.edge_select_vars[train][out_e] for out_e in self.train_graphs[train].out_edges(op)))

                    if obj["coeff"]:
                        self.model.add(self.threshold_vars[train, op] >= self.op_start_vars[train, op] - obj["threshold"]).OnlyEnforceIf(op_selected)
                    else:
                        self.model.add(self.op_start_vars[train, op] + 1 <= obj["threshold"]).OnlyEnforceIf([self.threshold_vars[train, op].Not(), op_selected])

                if train in self.semi_fixed:
                    op = obj["operation"]
                    if op not in list(self.feasible_sol[train].keys()):
                        continue
                    if obj["coeff"]:
                        self.model.add(self.threshold_vars[train, op] >= self.op_start_vars[train, op] - obj["threshold"])
                    else:
                        self.model.add(self.op_start_vars[train, op] + 1 <= obj["threshold"]).OnlyEnforceIf(self.threshold_vars[train, op].Not())



    def add_path_constraints(self):
        with TimeLogger("Path Constraints"):
            for train in self.choice:
                last_op = len(self.train_graphs[train].nodes) - 1

                self.model.add(sum(self.edge_select_vars[train][out_edge] for out_edge in self.train_graphs[train].out_edges(0)) == 1)
                self.model.add(sum(self.edge_select_vars[train][in_edge] for in_edge in self.train_graphs[train].in_edges(last_op)) == 1)

                for j in range(1, last_op):
                    in_edges = self.train_graphs[train].in_edges(j)
                    out_edges = self.train_graphs[train].out_edges(j)
                    self.model.add(sum(self.edge_select_vars[train][in_edge] for in_edge in in_edges) == sum(self.edge_select_vars[train][out_edge] for out_edge in out_edges))


    def add_timing_constraints(self):
        with TimeLogger("Timing Constraints"):
            for train in self.choice:
                for op in range(len(self.train_graphs[train].nodes)):
                    for in_edge in self.train_graphs[train].in_edges(op):
                        self.model.add(self.op_start_vars[train, in_edge[0]] + self.trains[train][in_edge[0]]["min_duration"] <= self.op_start_vars[train, op]).OnlyEnforceIf(self.edge_select_vars[train][in_edge])

                    for out_edge in self.train_graphs[train].out_edges(op):
                        self.model.add(self.op_end_vars[train, op] == self.op_start_vars[train, out_edge[1]]).OnlyEnforceIf(self.edge_select_vars[train][out_edge])

            for train in self.semi_fixed:
                ops = list(self.feasible_sol[train].keys())

                for i, (op, timings) in enumerate(self.feasible_sol[train].items()):
                    succ_op = ops[i + 1] if i + 1 < len(ops) else None
                    if succ_op is None:
                        self.model.add(self.op_end_vars[train, op] == self.op_start_vars[train, op] + self.trains[train][op]["min_duration"])
                    else:
                        self.model.add(self.op_start_vars[train, op] + self.trains[train][op]["min_duration"] <= self.op_start_vars[train, succ_op])
                        self.model.add(self.op_end_vars[train, op] == self.op_start_vars[train, succ_op])


    def add_resource_constraints(self):
        with TimeLogger("Resource Constraints"):
            for res, trains in self.resource_conflicts.items():
                if len(trains.keys()) == 1:
                    continue
                else:
                    interval_vars = []

                    for train, timings in trains.items():
                        if train in self.fix_trains:
                            interval_start = min(timings["start"])
                            interval_end = max(timings["end"])

                            rt = next((f_res["release_time"] for f_res in self.feasible_sol[train][interval_end]["resources"] if f_res["resource"] == res), None)
                            assert rt is not None, f"Releasetime for resource {res}, fixed train {train} not found"

                            start = self.feasible_sol[train][interval_start]["start"]
                            size = self.feasible_sol[train][interval_end]["end"] + rt - start
                            interval_vars.append(self.model.NewFixedSizeIntervalVar(start=start, size=size, name=""))
                        elif train in self.choice:
                            start_var = self.model.NewIntVar(lb=0, ub=2 ** 40, name="")
                            for start in timings["start"]:
                                start_selected = self.model.NewBoolVar(name="")
                                self.model.add(sum(self.edge_select_vars[train][out_e] for out_e in self.train_graphs[train].out_edges(start)) == start_selected)
                                self.model.add(start_var <= self.op_start_vars[train, start]).OnlyEnforceIf(start_selected)

                            end_var = self.model.NewIntVar(lb=0, ub=2 ** 40, name="")
                            for end in timings["end"]:
                                rt = self.find_release_time(train, end, res)
                                end_selected = self.model.NewBoolVar(name="")
                                self.model.add(sum(self.edge_select_vars[train][out_e] for out_e in self.train_graphs[train].out_edges(end)) == end_selected)
                                self.model.add(end_var >= self.op_end_vars[train, end] + rt).OnlyEnforceIf(end_selected)

                            res_avoidance_possible = res in self.resource_usage_vars[train].keys()
                            size = self.model.NewIntVar(lb=0, ub=2 ** 40, name="")

                            if res_avoidance_possible:
                                interval_vars.append(self.model.NewOptionalIntervalVar(start=start_var, end=end_var, is_present=self.resource_usage_vars[train][res], size=size, name=f"Optional interval for resource {res} for Train {train}"))
                            else:
                                interval_vars.append(self.model.NewIntervalVar(start=start_var, end=end_var, size=size, name=""))
                        elif train in self.semi_fixed:
                            size = self.model.NewIntVar(lb=0, ub=2 ** 40, name="")
                            end = self.model.NewIntVar(lb=0, ub=2 ** 40, name="")
                            rt = next((f_res["release_time"] for f_res in self.feasible_sol[train][max(timings["end"])]["resources"] if f_res["resource"] == res), None)
                            assert rt is not None, f"Releasetime for resource {res}, fixed train {train} not found"

                            self.model.add(end == rt + self.op_end_vars[train, max(timings["end"])])
                            interval_vars.append(self.model.NewIntervalVar(start=self.op_start_vars[train, min(timings["start"])], end=end, size=size, name=""))

                    self.model.AddNoOverlap(interval_vars)


    def add_resource_usage_constraints(self):
        for train in self.choice:
            for res, res_usage_var in self.resource_usage_vars[train].items():
                res_starts = self.resource_conflicts[res][train]["start"]
                in_edges = [in_e for s in res_starts for in_e in self.train_graphs[train].in_edges(s)]
                self.model.add(sum(self.edge_select_vars[train][in_e] for in_e in in_edges) == 0).OnlyEnforceIf(res_usage_var.Not())


    def add_deadlock_constraints(self):
        with TimeLogger("Deadlock Constraints"):
            now = time()
            connected_comps = list(nx.weakly_connected_components(self.deadlock_graph))
            for comp in connected_comps:
                subgraph = EventSorter.create_subgraph(self.deadlock_graph, comp)

                for cycle in nx.simple_cycles(subgraph, len(self.trains)):
                    if time() - now > 5:
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
        for train in self.choice:
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
                    if round(self.solver.value(self.edge_select_vars[train][(v, succ)])) == 1:
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

        for train in self.semi_fixed:
            for op, timings in self.feasible_sol[train].items():
                timings["start"] = round(self.solver.value(self.op_start_vars[train, op]))
                timings["end"] = round(self.solver.value(self.op_end_vars[train, op]))


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
        with TimeLogger("Deadlock Constraints"):
            graph = nx.DiGraph()
            critical_resources = [res["resource"] for i in self.choice for op in self.trains[i] for res in op["resources"]]

            for i, train in enumerate(self.trains):
                if i in self.choice:
                    for j, op in enumerate(train):
                        current_resources = [r["resource"] for r in op["resources"]]
                        for res in op["resources"]:
                            for succ in op["successors"]:
                                for succ_res in train[succ]["resources"]:
                                    edge = (res["resource"], succ_res["resource"])
                                    if edge[0] == edge[1] or succ_res["resource"] in current_resources or type(res["release_time"]) == int:
                                        continue
                                    else:
                                        if edge in graph.edges:
                                            edge_data = graph[edge[0]][edge[1]].get("data", [])
                                            edge_data.append((i, j))
                                            graph[edge[0]][edge[1]]["data"] = edge_data
                                        else:
                                            graph.add_nodes_from([res["resource"], succ_res["resource"]])
                                            graph.add_edge(edge[0], edge[1], data=[(i, j)])
                else:
                    succ_iterator = iter(self.feasible_sol[i].keys())
                    next(succ_iterator, None)
                    for k, (op, timings) in enumerate(self.feasible_sol[i].items()):
                        current_resources = [r["resource"] for r in timings["resources"]]
                        succ = next(succ_iterator, None)
                        for res in timings["resources"]:
                            if res["release_time"] != 0:
                                continue
                            for succ_res in self.feasible_sol[i][succ]["resources"]:
                                edge = (res["resource"], succ_res["resource"])
                                if edge[0] == edge[1] or succ_res["resource"] in current_resources or (edge[0] not in critical_resources) or (edge[1] not in critical_resources):
                                    continue
                                else:
                                    if edge in graph.edges:
                                        edge_data = graph[edge[0]][edge[1]].get("data", [])
                                        edge_data.append((i, op))
                                        graph[edge[0]][edge[1]]["data"] = edge_data
                                    else:
                                        graph.add_nodes_from([res["resource"], succ_res["resource"]])
                                        graph.add_edge(edge[0], edge[1], data=[(i, op)])

            return graph


    def create_train_graphs(self):
        train_graphs = []
        for train in self.trains:
            graph = nx.DiGraph()
            graph.add_nodes_from([i for i, _ in enumerate(train)])

            for i, operation in enumerate(train):
                graph.add_edges_from([(i, v) for v in operation["successors"]])
            train_graphs.append(graph)
        return train_graphs