import itertools, random
import networkx as nx
import time

from copy import deepcopy
from collections import defaultdict
from ortools.sat.python import cp_model as cp
from logger import Log


class TrainSolver:
    def __init__(self, instance, feasible_solution, fix_trains, choice, resource_evaluation, train_resource_usage, start):
        self.start = start
        self.choice = choice
        self.fix_trains = fix_trains
        self.feasible_solution = feasible_solution
        self.trains = instance.trains
        self.train_graphs = self.create_train_graphs()
        self.resource_evaluation = resource_evaluation
        self.train_resource_usage = train_resource_usage
        self.conflicted_resources = set()
        for train in self.choice:
            self.conflicted_resources.update(self.train_resource_usage[train])

        self.model = cp.CpModel()
        self.solver = cp.CpSolver()

        self.resource_conflicts = dict()  # A mapping from a resource to a list of operations using this operation
        self.op_start_vars = dict()
        self.op_end_vars = dict()
        self.edge_select_vars = list()
        self.resource_usage_vars = defaultdict(dict)

        for i in self.choice:
            select_vars = {}

            for res in self.train_resource_usage[i]:
                self.resource_usage_vars[i][res] = self.model.new_bool_var(name=f"Train {i} uses resource {res}")

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
                    if res["resource"] in self.conflicted_resources:
                        if res["resource"] in self.resource_conflicts.keys():
                            self.resource_conflicts[res["resource"]].append((j, op))
                        else:
                            self.resource_conflicts[res["resource"]] = [(j, op)]

        self.add_path_constraints()
        self.add_timing_constraints()
        self.add_resource_conflict_constraints()
        self.add_resource_usage_constraints()

        self.set_objective()


    def set_objective(self):
        self.model.minimize(sum(sum(self.resource_evaluation[res] * var for res, var in self.resource_usage_vars[i].items()) for i in self.choice))
        return


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

            self.model.add(self.op_end_vars[train, 0] >= self.start)
            last_op = len(self.trains[train]) - 1
            self.model.add(self.op_end_vars[train, last_op] == self.op_start_vars[train, last_op] + self.trains[train][last_op]["min_duration"])




    def add_resource_conflict_constraints(self):
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


    def add_resource_usage_constraints(self):
        for i, train in enumerate(self.choice):
            for j, op in enumerate(self.trains[train]):
                out_edges = self.train_graphs[train].out_edges(j)
                for res in op["resources"]:
                    self.model.add(sum(self.edge_select_vars[i][out_e] for out_e in out_edges) == 0).OnlyEnforceIf(self.resource_usage_vars[train][res["resource"]].Not())


    def solve(self):
        path_status = self.solver.Solve(self.model)

        if path_status == cp.OPTIMAL or path_status == cp.FEASIBLE:

            self.fix_path()
            self.model.clear_objective()
            self.model.minimize(sum(self.op_start_vars[choice, len(self.trains[choice]) - 1] for choice in self.choice))

            time_status = self.solver.Solve(self.model)

            if time_status == cp.OPTIMAL or time_status == cp.FEASIBLE:
                self.update_feasible_solution()
                return True
        return False


    def fix_path(self):
        for i, train in enumerate(self.choice):
            for j, op in enumerate(self.trains[train]):
                for succ in op["successors"]:
                    if round(self.solver.value(self.edge_select_vars[i][j, succ])):
                        self.model.add(self.edge_select_vars[i][j, succ] == 1)
        return


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


class Heuristic:
    def __init__(self, instance):
        self.log = Log(instance.objectives)
        self.instance = instance
        self.trains = instance.trains
        self.resource_appearances, self.train_to_resources = self.count_resource_appearances()
        self.start_graph = self.create_start_graph()
        self.blocking_dependencies = self.calculate_blocking_dependencies()
        self.increment_release_times()


    def schedule(self):
        feasible_solution = [{} for _ in range(len(self.trains))]
        scc_start = 0
        for scc in self.blocking_dependencies:
            scc_resource_evaluation = deepcopy(self.resource_appearances)
            scheduled_trains = []
            unscheduled_trains = deepcopy(scc)

            while len(unscheduled_trains):
                # print(f"{scheduled_trains} : {unscheduled_trains}")
                to_schedule = random.choices(unscheduled_trains, k=1)
                solution_found = TrainSolver(self.instance, feasible_solution, scheduled_trains, to_schedule, self.resource_appearances, self.train_to_resources, scc_start).solve()

                if not solution_found:
                    self.update_resource_appearances(to_schedule, scc_resource_evaluation)
                    conflicted_trains = self.calculate_conflicted_trains(to_schedule, scheduled_trains, scc, feasible_solution)

                    # Reschedule conflicted trains with punished resources
                    for train in conflicted_trains:
                        TrainSolver(self.instance, feasible_solution, [s for s in scheduled_trains if s != train], [train], self.resource_appearances, self.train_to_resources, scc_start).solve()

                    solution_found = TrainSolver(self.instance, feasible_solution, scheduled_trains, to_schedule, self.resource_appearances, self.train_to_resources, scc_start).solve()
                    if not solution_found:
                        conflicted_trains = self.calculate_conflicted_trains(to_schedule, scheduled_trains, scc, feasible_solution)
                        for t in to_schedule:
                            continue
                            # print(f"{t} is blocked by {[in_e[0] for in_e in self.start_graph.in_edges(t) if in_e[0] in scc]}")
                            # print(f"{t} blocks {[in_e[1] for in_e in self.start_graph.out_edges(t) if in_e[1] in scc]}")
                        conflicted_trains.sort(key=lambda x: feasible_solution[x][len(self.trains[x]) - 1]["start"])

                        for train in conflicted_trains:
                            scheduled_trains.remove(train)
                            unscheduled_trains.append(train)
                            solution_found = TrainSolver(self.instance, feasible_solution, scheduled_trains, to_schedule, self.resource_appearances, self.train_to_resources, scc_start).solve()

                            if solution_found:
                                break


                for train in to_schedule:
                    unscheduled_trains.remove(train)
                    scheduled_trains.extend(to_schedule)
                    scheduled_trains.sort()

            max_end = 0
            max_rt = 0
            for train in scc:
                for op, timings in feasible_solution[train].items():
                    max_end = max(max_end, timings["end"])
                    for res in timings["resources"]:
                        max_rt = max(max_rt, res["release_time"])

            scc_start = max_end + max_rt

        self.log.set_solution(feasible_solution)
        self.log.heuristic_calculation_time = time.time() - self.log.start
        return self.log


    def calculate_conflicted_trains(self, conflicting_trains, scheduled_trains, scc, feasible_solution):
        conflicted_trains = set()
        conflicting_resources = set()
        for conflicting_train in conflicting_trains:
            conflicting_resources.update(self.train_to_resources[conflicting_train])

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
                    if start_res in self.train_to_resources[t2]:
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


    def count_resource_appearances(self):
        resource_appearances = {}
        train_to_resources = {}

        for i, train in enumerate(self.trains):
            used_resources = set()
            for j, op in enumerate(train):
                for res in op["resources"]:
                    used_resources.add(res["resource"])
            for res in used_resources:
                if resource_appearances.get(res) is not None:
                    resource_appearances[res] += 1
                else:
                    resource_appearances[res] = 1
            train_to_resources[i] = used_resources
        return resource_appearances, train_to_resources


    def update_resource_appearances(self, to_schedule, scc_resource_evaluation):
        max_counter = max(scc_resource_evaluation.values())
        used_resources = set()
        for train in to_schedule:
            used_resources.update(self.train_to_resources[train])
        for res in used_resources:
            scc_resource_evaluation[res] += max_counter


    def increment_release_times(self):
        for train in self.instance.trains:
            for op in train:
                for res in op["resources"]:
                    if res["release_time"] == 0:
                        res["release_time"] = 1