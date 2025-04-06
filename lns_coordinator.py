import itertools
import random

from LNS_displib_solver import LnsDisplibSolver
from time import time
from logger import Log

import copy

class LnsCoordinator:
    def __init__(self, instance, feasible_sol, time_limit):
        self.instance = instance
        self.feasible_sol = copy.deepcopy(feasible_sol)
        self.objective = calculate_objective_value(self.instance.objectives, self.feasible_sol)

        self.start_time = time()
        self.has_to_be_finished_at = self.start_time + time_limit

        self.log = Log(self.feasible_sol, self.instance.objectives)

        self.train_usage = {train: 0 for train in range(len(self.instance.trains))}
        self.current_strategy = random.randint(0, 4)


    def solve(self, alpha=0.1, beta=0.1, initial_weight=1):
        # TODO: Maybe change the weights up in the future...
        strategy_weights = {"conflict": initial_weight, "objective": initial_weight, "random": initial_weight, "least_used": initial_weight, "nearest_threshold": initial_weight}
        size_weights = {size: (1 / size) ** 2 for size in range(1, len(self.instance.trains) + 1)}

        strategy_functions = {"conflict": lambda s: self.choose_resource_conflicted_trains(s),
                              "objective": lambda s: self.choose_strong_overall_delay(s),
                              "random": lambda s: self.choose_random_trains(s),
                              "least_used": lambda s: self.choose_least_used(s),
                              "nearest_threshold": lambda s: self.choose_nearest_threshold(s)}

        # actually, 5 seconds might be enough
        while self.calculate_remaining_time() > 5:
            strategy_W, size_W = sum(w for w in strategy_weights.values()), sum(w for w in size_weights.values())

            strategy = random.choices(["conflict", "objective", "random", "least_used", "nearest_threshold"], weights= [w / strategy_W for w in strategy_weights.values()], k=1)[0]
            size = random.choices([size for size in range(1, len(size_weights.keys()) + 1)], weights=[w / size_W for w in size_weights.values()], k=1)[0]

            choice = strategy_functions[strategy](size)
            for train in choice:
                self.train_usage[train] += 1

            new_feasible_sol = LnsDisplibSolver(self.instance, self.feasible_sol, choice, self.calculate_remaining_time() - 2).solve()
            new_objective_value = calculate_objective_value(self.instance.objectives, new_feasible_sol)

            if new_objective_value == 0:
                self.objective = new_objective_value
                self.feasible_sol = new_feasible_sol
                self.log.set_solution(self.feasible_sol)
                break

            if new_objective_value < self.objective:
                print(f"Found solution with better objective {new_objective_value} by rescheduling {choice} increasing <{strategy}> by {alpha * (self.objective / new_objective_value)}")

                strategy_weights[strategy] = strategy_weights[strategy] + alpha * (self.objective / new_objective_value)
                size_weights[size] = size_weights[size] + alpha * (self.objective / new_objective_value)

                self.objective = new_objective_value
                self.feasible_sol = new_feasible_sol
                self.log.set_solution(self.feasible_sol)
            elif new_objective_value == self.objective:
                print(f"Found solution with same objective {new_objective_value} by rescheduling {choice} decreasing <{strategy}> by {strategy_weights[strategy] - strategy_weights[strategy] * (1 - beta)}")

                # Not finding better solutions should be punished
                strategy_weights[strategy] = strategy_weights[strategy] * (1 - beta)
                size_weights[size] = size_weights[size] * (1 - beta)

                # Even though the solution isn't better, take it for further use. This might help us getting out of local optima
                self.objective = new_objective_value
                self.feasible_sol = new_feasible_sol
                self.log.set_solution(self.feasible_sol)
            else:
                print(f"Found solution with worse objective after rescheduling {choice} with <{strategy}>. Discarding solution.")
                # But returning a worse solution should be punished way harder
                strategy_weights[strategy] = strategy_weights[strategy] * (1 - beta) / 4
                size_weights[size] = size_weights[size] * (1 - beta) / 4

        print(f"The weights were <{strategy_weights}> : <{size_weights.items()}>")



    def calculate_remaining_time(self):
        return max(0, self.has_to_be_finished_at - time())


    def choose_least_used(self, size):
        choice = sorted(self.train_usage, key=self.train_usage.get)[:min(len(self.feasible_sol), size)]
        return sorted(choice)  # error handling (size may be invalid)


    def choose_random_trains(self, size):
        return sorted(random.sample([i for i in range(len(self.feasible_sol))], k=size))


    def choose_resource_conflicted_trains(self, size):
        resources_per_train = {i: set() for i in range(len(self.feasible_sol))}
        used_resources = set()

        for i, train in enumerate(self.feasible_sol):
            for op, timings in train.items():
                for res in timings["resources"]:
                    resources_per_train[i].add(res["resource"])
                    used_resources.add(res["resource"])

        choice = None
        conflicts = 0
        for comb in itertools.combinations([i for i in range(len(self.feasible_sol))], size):
            same_resources = used_resources
            for train in comb:
                same_resources = same_resources.intersection(resources_per_train[train])

            if not choice or len(same_resources) > conflicts:
                choice = comb
                conflicts = len(same_resources)
            if len(used_resources) == conflicts:
                # We can use another condition too to reduce duration, if the combination is good enough
                break

        return sorted(choice)


    def choose_nearest_threshold(self, size):
        threshold_overshoot_per_train = {i: 2 ** 40 for i in range(len(self.feasible_sol))} # Not very smart but will do for now

        for obj in self.instance.objectives:
            train = obj["train"]
            op = obj["operation"]

            if self.feasible_sol[train].get(op):
                diff_to_threshold = max(0, self.feasible_sol[train][op]["start"] - obj["threshold"])
                threshold_overshoot_per_train[train] = min(threshold_overshoot_per_train[train], diff_to_threshold)

        choice = sorted(threshold_overshoot_per_train, key=threshold_overshoot_per_train.get)[:min(len(self.feasible_sol), size)]
        return sorted(choice)


    def choose_strong_linear_delay(self, size):
        # To be implemented
        pass


    def choose_strong_increment_delay(self, size):
        # To be implemented
        pass


    def choose_strong_overall_delay(self, size):
        objective_per_train = {i: 0 for i in range(len(self.feasible_sol))}

        for obj in self.instance.objectives:
            train = obj["train"]
            op = obj["operation"]

            if op in self.feasible_sol[train].keys():
                val = objective_per_train[train] + obj["coeff"] * max(0, self.feasible_sol[train][op]["start"] - obj["threshold"]) + obj["increment"] * (self.feasible_sol[train][op]["start"] - obj["threshold"] > 0) # Sometimes, it does not save the new result...
                objective_per_train[train] += val

        choice = sorted(objective_per_train, key=objective_per_train.get, reverse=True)[:min(len(self.feasible_sol), size)]
        return sorted(choice) # error handling (size may be invalid)


def calculate_objective_value(objectives, new_feasible_sol):
    value = 0
    for obj in objectives:
        train = obj["train"]
        op = obj["operation"]

        if op in list(new_feasible_sol[train].keys()):
            value += (obj["coeff"] * max(0, new_feasible_sol[train][op]["start"] - obj["threshold"]) + obj["increment"] * (1 if new_feasible_sol[train][op]["start"] >= obj["threshold"] else 0))

    return value