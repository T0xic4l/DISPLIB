import random
import logging
import networkx as nx

from copy import deepcopy
from collections import defaultdict
from itertools import count

from ortools.sat.python import cp_model as cp
from logger import TimeLogger


class TrainSolver:
    def __init__(self, instance, feasible_solution, fix_trains, choice, resource_evaluation, semi_fixed_trains, train_resource_usage, start_time, single_train_schedule=False):
        with TimeLogger("Model Setup"):
            self.start_time = start_time
            self.single_train_schedule = single_train_schedule
            self.choice = choice
            self.fix_trains = fix_trains
            self.semi_fixed_trains = semi_fixed_trains
            self.feasible_solution = feasible_solution
            self.objectives = instance.objectives
            self.trains = instance.trains
            self.train_graphs = [create_train_graph(train) for train in self.trains]
            self.resource_evaluation = resource_evaluation
            self.train_resource_usage = train_resource_usage

            self.model = cp.CpModel()
            self.solver = cp.CpSolver()

            self.resource_conflicts = self.calculate_resource_conflicts() # [res][train]["start" / "end"]
            self.op_start_vars = dict()
            self.op_end_vars = dict()
            self.edge_select_vars = defaultdict(dict)
            self.resource_usage_vars = defaultdict(dict)
            self.threshold_vars = dict()

            self.create_variables()

            self.add_path_constraints()
            self.add_timing_constraints()
            self.add_resource_conflict_constraints()
            self.add_resource_usage_constraints()
            if self.single_train_schedule:
                self.add_threshold_constraints()
            else:
                pass


    def solve(self):
        with TimeLogger("Solving"):
            self.solver.parameters.log_search_progress = False
            self.solver.parameters.num_workers = 8

            if self.single_train_schedule:
                self.set_time_objective()
                status = self.solver.Solve(self.model)

                if status == cp.OPTIMAL or status == cp.FEASIBLE:
                    self.update_feasible_solution()
                    return True
                elif status == cp.INFEASIBLE:
                    logging.warning("Model is infeasible.")
                    return False

            self.set_path_objective()
            path_status = self.solver.Solve(self.model)

            if path_status == cp.OPTIMAL or path_status == cp.FEASIBLE:
                self.fix_path()
                self.model.clear_objective()
                self.set_time_objective()

                time_status = self.solver.Solve(self.model)

                if time_status == cp.OPTIMAL or time_status == cp.FEASIBLE:
                    self.update_feasible_solution()
                    return True
            return False


    def set_path_objective(self):
        self.solver.parameters.max_time_in_seconds = 8
        self.model.minimize(sum(sum(self.resource_evaluation[res] * var for res, var in self.resource_usage_vars[i].items()) for i in self.choice))


    def set_time_objective(self):
        self.solver.parameters.max_time_in_seconds = 2
        self.model.minimize(sum(self.op_start_vars[i, len(self.trains[i]) - 1] for i in self.choice + self.semi_fixed_trains))


    def add_threshold_constraints(self):
        with TimeLogger("Threshold Constraints"):
            for obj in self.objectives:
                train = obj["train"]

                if train in self.choice:
                    op = obj["operation"]
                    op_selected = self.model.new_bool_var(name="")
                    if op != 0:
                        self.model.add(op_selected == sum(
                            self.edge_select_vars[train][in_e] for in_e in self.train_graphs[train].in_edges(op)))
                    else:
                        self.model.add(op_selected == sum(
                            self.edge_select_vars[train][out_e] for out_e in self.train_graphs[train].out_edges(op)))

                    if obj["coeff"]:
                        self.model.add(self.threshold_vars[train, op] >= self.op_start_vars[train, op] - obj["threshold"]).OnlyEnforceIf(op_selected)
                    else:
                        self.model.add(self.op_start_vars[train, op] + 1 <= obj["threshold"]).OnlyEnforceIf([self.threshold_vars[train, op].Not(), op_selected])


    def create_objective_vars(self):
        for obj in self.objectives:
            if obj["train"] in self.choice:
                if obj["coeff"] != 0:
                    self.threshold_vars[obj["train"], obj["operation"]] = self.model.NewIntVar(lb=0, ub=2 ** 40, name="")
                else:
                    self.threshold_vars[obj["train"], obj["operation"]] = self.model.NewBoolVar(name="")


    def set_obj_objective(self):
        self.model.minimize(sum(obj["coeff"] * self.threshold_vars[obj["train"], obj["operation"]] + obj["increment"] * self.threshold_vars[obj["train"], obj["operation"]] for obj in self.objectives if obj["train"] in self.choice))


    def create_variables(self):
        for i in self.choice:
            for res in self.train_resource_usage[i]:
                if check_resource_avoidance(self.trains[i], [res], return_path=False):
                    self.resource_usage_vars[i][res] = self.model.NewBoolVar(name="")

            for j, op in enumerate(self.trains[i]):
                self.op_start_vars[i, j] = self.model.NewIntVar(lb=op["start_lb"], ub=op["start_ub"], name="")
                self.op_end_vars[i, j] = self.model.NewIntVar(lb=op["start_lb"] + op["min_duration"], ub=2 ** 40, name="")

                for s in op["successors"]:
                    self.edge_select_vars[i][j, s] = self.model.NewBoolVar(name="")

        for i in self.semi_fixed_trains:
            for op, timings in self.feasible_solution[i].items():
                self.op_start_vars[i, op] = self.model.NewIntVar(lb=self.trains[i][op]["start_lb"], ub=self.trains[i][op]["start_ub"], name="")
                self.op_end_vars[i, op] = self.model.NewIntVar(lb=self.trains[i][op]["start_lb"] + self.trains[i][op]["min_duration"], ub=2 ** 40, name="")

        if self.single_train_schedule:
            self.create_objective_vars()


    def add_path_constraints(self):
        for train in self.choice:
            last_op = len(self.train_graphs[train].nodes) - 1

            self.model.add(sum(self.edge_select_vars[train][out_edge] for out_edge in self.train_graphs[train].out_edges(0)) == 1)
            self.model.add(sum(self.edge_select_vars[train][in_edge] for in_edge in self.train_graphs[train].in_edges(last_op)) == 1)

            for j in range(1, last_op):
                in_edges = self.train_graphs[train].in_edges(j)
                out_edges = self.train_graphs[train].out_edges(j)
                self.model.add(sum(self.edge_select_vars[train][in_edge] for in_edge in in_edges) == sum(self.edge_select_vars[train][out_edge] for out_edge in out_edges))


    def add_timing_constraints(self):
        for train in self.choice:
            for op in range(len(self.train_graphs[train].nodes)):
                for in_edge in self.train_graphs[train].in_edges(op):
                    self.model.add(self.op_start_vars[train, in_edge[0]] + self.trains[train][in_edge[0]]["min_duration"] <= self.op_start_vars[train, op]).OnlyEnforceIf(self.edge_select_vars[train][in_edge])

                for out_edge in self.train_graphs[train].out_edges(op):
                    self.model.add(self.op_end_vars[train, op] == self.op_start_vars[train, out_edge[1]]).OnlyEnforceIf(self.edge_select_vars[train][out_edge])

            self.model.add(self.op_end_vars[train, 0] >= self.start_time)
            last_op = len(self.trains[train]) - 1
            self.model.add(self.op_end_vars[train, last_op] == self.op_start_vars[train, last_op] + self.trains[train][last_op]["min_duration"])

        for train in self.semi_fixed_trains:
            ops = list(self.feasible_solution[train].keys())

            for i, (op, timings) in enumerate(self.feasible_solution[train].items()):
                succ_op = ops[i + 1] if i + 1 < len(ops) else None
                if succ_op is None:
                    self.model.add(self.op_end_vars[train, op] == self.op_start_vars[train, op] + self.trains[train][op]["min_duration"])
                else:
                    self.model.add(self.op_start_vars[train, op] + self.trains[train][op]["min_duration"] <= self.op_start_vars[train, succ_op])
                    self.model.add(self.op_end_vars[train, op] == self.op_start_vars[train, succ_op])
            self.model.add(self.op_end_vars[train, 0] >= self.start_time)


    def add_resource_conflict_constraints(self):
        for res, trains in self.resource_conflicts.items():
            if len(trains.keys()) == 1:
                continue
            else:
                interval_vars = []

                for train, timings in trains.items():
                    if train in self.fix_trains:
                        interval_start = min(timings["start"])
                        interval_end = max(timings["end"])

                        rt = next((f_res["release_time"] for f_res in self.feasible_solution[train][timings["end"][0]]["resources"] if f_res["resource"] == res), None)
                        assert rt is not None, f"Releasetime for resource {res}, fixed train {train} not found"

                        start = self.feasible_solution[train][interval_start]["start"]
                        size = self.feasible_solution[train][interval_end]["end"] + rt - start
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
                            interval_vars.append(self.model.NewOptionalIntervalVar(start=start_var, end=end_var, is_present=self.resource_usage_vars[train][res], size=size, name=""))
                        else:
                            interval_vars.append(self.model.NewIntervalVar(start=start_var, end=end_var, size=size, name=""))
                    elif train in self.semi_fixed_trains:
                        size = self.model.new_int_var(lb=0, ub= 2 ** 40, name="")
                        end = self.model.new_int_var(lb=0, ub= 2 ** 40, name="")
                        rt = next((f_res["release_time"] for f_res in self.feasible_solution[train][timings["end"][0]]["resources"] if f_res["resource"] == res), None)
                        assert rt is not None and rt != 0, f"Releasetime for resource {res}, semi-fixed train {train} not found"
                        self.model.add(end == rt + self.op_end_vars[train, max(timings["end"])])

                        interval_vars.append(self.model.NewIntervalVar(start=self.op_start_vars[train, timings["start"][0]], end=end, size=size, name=""))
                self.model.AddNoOverlap(interval_vars)


    def add_resource_usage_constraints(self):
        for train in self.choice:
            for res, res_usage_var in self.resource_usage_vars[train].items():
                res_starts = self.resource_conflicts[res][train]["start"]
                in_edges = [in_e for s in res_starts for in_e in self.train_graphs[train].in_edges(s)]
                self.model.add(sum(self.edge_select_vars[train][in_e] for in_e in in_edges) == 0).OnlyEnforceIf(res_usage_var.Not())


    def fix_path(self):
        for train in self.choice:
            for j, op in enumerate(self.trains[train]):
                for succ in op["successors"]:
                    if round(self.solver.value(self.edge_select_vars[train][j, succ])):
                        self.model.add(self.edge_select_vars[train][j, succ] == 1)


    def calculate_resource_conflicts(self):
        # dict[res][train]["start"/"end"]
        new_resource_conflicts = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        # get start- and end operations for variable trains, there might be more than one
        for i in self.choice:
            res_to_ops = defaultdict()
            for j, op in enumerate(self.trains[i]):
                for res in op["resources"]:
                    if res_to_ops.get(res["resource"]) is not None:
                        res_to_ops[res["resource"]].append(j)
                    else:
                        res_to_ops[res["resource"]] = [j]

            for res in self.train_resource_usage[i]:
                start_ops = deepcopy(res_to_ops[res])
                end_ops = deepcopy(res_to_ops[res])

                for op in res_to_ops[res]:
                    if all(any(succ_res["resource"] == res for succ_res in self.trains[i][succ]["resources"]) for succ in self.trains[i][op]["successors"]):
                        end_ops.remove(op)

                    if op != 0:
                        predecessors = [pred[0] for pred in self.train_graphs[i].in_edges(op)]
                        if all(any(pred_res["resource"] == res for pred_res in self.trains[i][pred]["resources"]) for pred in predecessors):
                            start_ops.remove(op)

                new_resource_conflicts[res][i]["start"] = start_ops
                new_resource_conflicts[res][i]["end"] = end_ops

        for i in self.semi_fixed_trains:
            for op, timings in self.feasible_solution[i].items():
                for res in timings["resources"]:
                    if not new_resource_conflicts.get(res["resource"]):
                        new_resource_conflicts[res["resource"]][i]["start"] = []
                        new_resource_conflicts[res["resource"]][i]["end"] = []


        # get start- and end operation for fix trains - only one
        for i in self.fix_trains + self.semi_fixed_trains:
            pred = -1
            track_resources = list()
            for op, timings in self.feasible_solution[i].items():
                op_res = [res["resource"] for res in timings["resources"]]
                for res in op_res:
                    if res not in track_resources and new_resource_conflicts.get(res) is not None:
                        track_resources.append(res)
                        new_resource_conflicts[res][i]["start"].append(op)
                if pred != -1:
                    track_copy = deepcopy(track_resources)
                    for res in track_copy:
                        if res not in op_res:
                            track_resources.remove(res)
                            new_resource_conflicts[res][i]["end"].append(pred)
                pred = op
        return new_resource_conflicts


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
        for train in self.choice:
            train_sol = {}
            v = 0

            resource_list = []
            for res in self.trains[train][v]["resources"]:
                resource_list.append(res)

            train_sol.update({v: {"start": round(self.solver.value(self.op_start_vars[train, v])), "end": round(self.solver.value(self.op_end_vars[train, v])), "resources": resource_list}})

            while v != len(self.train_graphs[train].nodes) - 1:
                for succ in self.trains[train][v]["successors"]:
                    if round(self.solver.value(self.edge_select_vars[train][(v, succ)])) == 1:
                        v = succ

                        resource_list = []
                        for res in self.trains[train][v]["resources"]:
                            resource_list.append(res)

                        train_sol.update({v: {"start": round(self.solver.value(self.op_start_vars[train, v])), "end": round(self.solver.value(self.op_end_vars[train, v])), "resources": resource_list}})
                        break

            self.feasible_solution[train] = train_sol

        for train in self.semi_fixed_trains:
            for op, timings in self.feasible_solution[train].items():
                timings["start"] = round(self.solver.value(self.op_start_vars[train, op]))
                timings["end"] = round(self.solver.value(self.op_end_vars[train, op]))
            # logging.info(f"Feasible Sol for train {train}\n: {json.dumps(self.feasible_solution[train])}\n")


class Heuristic:
    def __init__(self, instance, res_eval, train_to_res):
        self.instance = instance
        self.trains = instance.trains
        self.res_eval = res_eval
        self.train_to_res = train_to_res

        self.start_graph = self.create_start_graph()
        self.blocking_dependencies = self.calculate_blocking_dependencies()
        self.increment_release_times()

        self.restricted_trains_to_path = dict()
        self.split_blocking_dependencies()


    def split_blocking_dependencies(self):
        for scc in self.blocking_dependencies:
            if not len(scc) == 1:
                self.split_scc(scc)


    def split_scc(self, scc):
        scc_graph = self.create_sub_graph(scc, self.start_graph)

        for node in list(scc_graph.nodes):
            blocking_trains = [in_edge[0] for in_edge in list(scc_graph.in_edges(node))]
            start_resources = [res["resource"] for train in blocking_trains for res in self.trains[train][0]["resources"]]
            path = check_resource_avoidance(self.trains[node], start_resources, return_path=True)
            if path:
                # After removing node, graph might be broken into more than one scc again
                self.restricted_trains_to_path[node] = path
                scc_graph.remove_node(node)
                condensed_graph = self.create_condensed_graph(scc_graph)
                blocking_dependency = list(nx.topological_sort(condensed_graph))

                # Overwrite Scc by sub_sccs
                scc_index = self.blocking_dependencies.index(scc)
                self.blocking_dependencies = (
                        self.blocking_dependencies[:scc_index] +
                        [[node]] + [list(condensed_graph.nodes[scc]["members"]) for scc in blocking_dependency] +
                        self.blocking_dependencies[scc_index + 1:])

                for condensed_node in blocking_dependency:
                    self.split_scc(list(condensed_graph.nodes[condensed_node]["members"]))
                break


    def schedule(self):
        logging.debug("Initial resource weights: %s", ", ".join(str(v) for v in self.res_eval.values()))

        feasible_solution = [{} for _ in range(len(self.trains))]
        scc_start = 0
        scheduled_sccs = []

        for scc in self.blocking_dependencies:
            logging.info(f"Start of scheduling scc: {scc}")

            scc_resource_evaluation = deepcopy(self.res_eval)
            scheduled_trains = []
            unscheduled_trains = scc.copy()

            if len(scc) == 1:
                feasible_solution[scc[0]] = self.schedule_single_train(scc[0], scc_start)
                if not TrainSolver(self.instance, feasible_solution, scheduled_sccs, scc, scc_resource_evaluation, [], self.train_to_res, 0, True).solve():
                    logging.warning(f"Could not schedule Train {scc[0]} with the solver. Scheduling it sequentially instead")
                    feasible_solution[scc[0]] = self.schedule_single_train(scc[0], scc_start)
            else:
                counter = count(start=1, step=1)

                while len(unscheduled_trains):
                    i = next(counter)

                    to_schedule = random.sample(unscheduled_trains, k=min(1, len(unscheduled_trains)))
                    conflicting_trains = self.calculate_conflicted_trains(to_schedule, scheduled_trains, scc, feasible_solution)
                    semi_fixed_trains = random.sample(conflicting_trains, min(4, len(conflicting_trains)))
                    fixed_trains = [train for train in scheduled_trains if train not in semi_fixed_trains]

                    logging.info(f"Iteration {i} - to schedule: {to_schedule} --- scheduled: {scheduled_trains} --- unscheduled: {unscheduled_trains}")
                    if not TrainSolver(self.instance, feasible_solution, fixed_trains, to_schedule, scc_resource_evaluation, semi_fixed_trains, self.train_to_res, scc_start).solve():
                        self.update_resource_appearances(to_schedule, scc_resource_evaluation)

                        random.shuffle(conflicting_trains)

                        for train in conflicting_trains:
                            if train in semi_fixed_trains:
                                semi_fixed_trains.remove(train)
                            else:
                                fixed_trains.remove(train)
                            scheduled_trains.remove(train)
                            unscheduled_trains.append(train)

                            if TrainSolver(self.instance, feasible_solution, fixed_trains, to_schedule, scc_resource_evaluation, semi_fixed_trains, self.train_to_res, scc_start).solve():
                                break
                            else:
                                self.update_resource_appearances(to_schedule, scc_resource_evaluation)

                    self.update_resource_appearances(to_schedule, scc_resource_evaluation, True)

                    logging.debug("Updated resource evaluation: %s", ", ".join(str(v) for v in scc_resource_evaluation.values()))

                    for train in to_schedule:
                        unscheduled_trains.remove(train)
                        scheduled_trains.append(train)

            max_end = 0
            max_rt = 0
            for train in scc:
                for op, timings in feasible_solution[train].items():
                    max_end = max(max_end, timings["end"])
                    for res in timings["resources"]:
                        max_rt = max(max_rt, res["release_time"])

            scc_start = max_end + max_rt
            logging.info(f"Scheduling {scc} done \n")
            scheduled_sccs.extend(scc)

        result = {
            "solution" : feasible_solution,
            "scc_count" : len(self.blocking_dependencies)
        }
        return result


    def calculate_conflicted_trains(self, conflicting_trains, scheduled_trains, scc, feasible_solution):
        conflicted_trains = set()
        conflicting_resources = set()
        for conflicting_train in conflicting_trains:
            conflicting_resources.update(self.train_to_res[conflicting_train])

        for train in scc:
            if train in conflicting_trains or train not in scheduled_trains:
                continue
            for op, timings in feasible_solution[train].items():
                for res in timings["resources"]:
                    if res["resource"] in conflicting_resources:
                        conflicted_trains.add(train)
        return list(conflicted_trains)


    def calculate_blocking_dependencies(self):
        start_graph = self.create_start_graph()
        condensed_graph = self.create_condensed_graph(start_graph)
        return [list(condensed_graph.nodes[scc]["members"]) for scc in list(nx.topological_sort(condensed_graph))]


    def create_start_graph(self):
        start_graph = nx.DiGraph()
        start_graph.add_nodes_from(range(len(self.trains)))

        for t1, blocking_train in enumerate(self.trains):
            start_resources = [res["resource"] for res in blocking_train[0]["resources"]]
            for t2, blocked_train in enumerate(self.trains):
                if t1 == t2:
                    continue
                for start_res in start_resources:
                    if start_res in self.train_to_res[t2]:
                        start_graph.add_edge(t1, t2)
        return start_graph


    @staticmethod
    def create_condensed_graph(start_graph):
        condensed_graph = nx.DiGraph()
        sccs = list(nx.strongly_connected_components(start_graph))
        node_to_scc = {}

        for i, scc in enumerate(sccs):
            for node in scc:
                node_to_scc[node] = i
            condensed_graph.add_node(i, members=scc)

        for u, v in start_graph.edges:
            scc_u = node_to_scc[u]
            scc_v = node_to_scc[v]
            if scc_u != scc_v:
                condensed_graph.add_edge(scc_u, scc_v)

        return condensed_graph


    @staticmethod
    def create_sub_graph(sub_nodes, graph):
        sub_graph = nx.DiGraph()
        for u, v in graph.edges:
            if u in sub_nodes and v in sub_nodes:
                sub_graph.add_edge(u, v)
        return sub_graph


    def update_resource_appearances(self, to_schedule, scc_resource_evaluation, decay = False):
        if decay:
            for res in scc_resource_evaluation.keys():
                scc_resource_evaluation[res] = max(round(scc_resource_evaluation[res] * 0.9, 2), 1)
        else:
            used_resources = set()
            for train in to_schedule:
                used_resources.update(self.train_to_res[train])

            for res in used_resources:
                scc_resource_evaluation[res] = min(max(10, round(scc_resource_evaluation[res] * 2, 2)), 100000)


    def increment_release_times(self):
        for train in self.instance.trains:
            for op in train:
                for res in op["resources"]:
                    if res["release_time"] == 0:
                        res["release_time"] = 1


    def schedule_single_train(self, train_id, start):
        current_time = start
        train = self.trains[train_id]
        train_solution = {}

        op = 0
        train_solution.update({0: {"start": 0, "end": max(train[0]["min_duration"], current_time), "resources": train[op]["resources"]}})
        current_time = train_solution[0]["end"]

        if train_id in self.restricted_trains_to_path.keys():
            del self.restricted_trains_to_path[train_id][0]
            for op in self.restricted_trains_to_path[train_id]:
                current_time = max(current_time, train[op]["start_lb"])
                train_solution.update({op: {"start": current_time, "end": current_time + train[op]["min_duration"], "resources": train[op]["resources"]}})
                current_time = train_solution[op]["end"]
        else:
            while op != len(train) - 1:
                op = train[op]["successors"][0]
                current_time = max(current_time, train[op]["start_lb"])
                train_solution.update({op: {"start": current_time, "end": current_time + train[op]["min_duration"], "resources": train[op]["resources"]}})
                current_time = train_solution[op]["end"]

        return train_solution


def create_train_graph(train):
    graph = nx.DiGraph()
    graph.add_nodes_from([i for i, _ in enumerate(train)])
    for i, operation in enumerate(train):
        graph.add_edges_from([(i, v) for v in operation["successors"]])
    return graph


def check_resource_avoidance(train, resources, return_path = False):
    start_res = [res["resource"] for res in train[0]["resources"]]
    if set(resources).intersection(start_res):
        return None

    train_graph = create_train_graph(train)
    for i, op in enumerate(train):
        for res in train[i]["resources"]:
            if res["resource"] in resources:
                train_graph.remove_node(i)
                break

    if not return_path:
        return nx.has_path(train_graph, source=0, target=len(train) - 1)
    else:
        try:
            return nx.shortest_path(train_graph, source=0, target=len(train) - 1)
        except nx.NetworkXNoPath:
            return None