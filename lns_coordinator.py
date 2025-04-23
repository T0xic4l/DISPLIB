import random, functools, logging

from LNS_displib_solver import LnsDisplibSolver
from time import time
from copy import deepcopy
from logger import TimeLogger

class LnsCoordinator:
    def __init__(self, instance, log, resource_appearances, train_to_resources, time_limit):
        self.instance = instance
        self.feasible_sol = log.get_solution()
        self.objective = calculate_objective_value(self.instance.objectives, self.feasible_sol)
        self.resource_appearances = resource_appearances # Not used ATM
        self.train_to_resources = train_to_resources

        self.start_time = time()
        self.has_to_be_finished_at = self.start_time + time_limit

        self.log = log

        self.train_usage = {train: 0 for train in range(len(self.instance.trains))}
        self.current_strategy = random.randint(0, 4)


    def solve(self):
        logging.info("Starting LNS...\n")

        strategy = 0
        size = 2

        strategy_functions = [lambda s: self.choose_strong_overall_delay(s),
                              lambda s: self.choose_resource_conflicted_trains(s),
                              lambda s: self.choose_random_trains(s),
                              lambda s: self.choose_least_used(s),
                              lambda s: self.choose_nearest_threshold(s)]

        # actually, 5 seconds might be enough
        while self.calculate_remaining_time() > 5:
            choice = strategy_functions[strategy % len(strategy_functions)](size)
            for train in choice:
                self.train_usage[train] += 1

            now = time()
            new_feasible_sol = LnsDisplibSolver(self.instance, self.feasible_sol, choice, deepcopy(self.train_to_resources), self.calculate_remaining_time() - 2, 2).solve()
            if time() - now < 25:
                size = min(size + 1, len(self.instance.trains))
            else:
                size = max(1, size - 1)

            new_objective_value = calculate_objective_value(self.instance.objectives, new_feasible_sol)

            if new_objective_value == 0:
                self.objective = new_objective_value
                self.feasible_sol = new_feasible_sol
                self.log.set_solution(self.feasible_sol)
                break

            if new_objective_value < self.objective:
                print(f"Found solution with better objective {new_objective_value} by rescheduling {choice} with <{strategy}>")

                self.objective = new_objective_value
                self.feasible_sol = new_feasible_sol
                self.log.set_solution(self.feasible_sol)
            elif new_objective_value == self.objective:
                print(f"Found solution with same objective {new_objective_value} by rescheduling {choice} decreasing <{strategy}> with {strategy}")
                strategy += 1

                # Even though the solution isn't better, take it for further use. This might help us getting out of local optima
                self.objective = new_objective_value
                self.feasible_sol = new_feasible_sol
                self.log.set_solution(self.feasible_sol)
            else:
                print(f"Found solution with worse objective after rescheduling {choice} with <{strategy}>. Discarding solution.")
                strategy += 1




    def calculate_remaining_time(self):
        return max(0, self.has_to_be_finished_at - time())


    def choose_least_used(self, size):
        choice = sorted(self.train_usage, key=self.train_usage.get)[:min(len(self.feasible_sol), size)]
        return choice  # error handling (size may be invalid)


    def choose_random_trains(self, size):
        return random.sample([i for i in range(len(self.feasible_sol))], k=size)


    def choose_resource_conflicted_trains(self, size):
        best_bound = 0
        sol = []
        now = time()

        while time() - now <= 1:
            trains = random.sample(list(self.train_to_resources.keys()), size)
            cardinality = len(functools.reduce(set.intersection, (self.train_to_resources[i] for i in trains))) > best_bound
            if cardinality > best_bound:
                best_bound = cardinality
                sol = trains

        return sol


    def choose_nearest_threshold(self, size):
        threshold_overshoot_per_train = {i: 2 ** 40 for i in range(len(self.feasible_sol))} # Not very smart but will do for now

        for obj in self.instance.objectives:
            train = obj["train"]
            op = obj["operation"]

            if self.feasible_sol[train].get(op):
                diff_to_threshold = max(0, self.feasible_sol[train][op]["start"] - obj["threshold"])
                threshold_overshoot_per_train[train] = min(threshold_overshoot_per_train[train], diff_to_threshold)

        choice = sorted(threshold_overshoot_per_train, key=threshold_overshoot_per_train.get)[:min(len(self.feasible_sol), size)]
        return choice


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
        return choice # error handling (size may be invalid)


def calculate_objective_value(objectives, new_feasible_sol):
    value = 0
    for obj in objectives:
        train = obj["train"]
        op = obj["operation"]

        if op in list(new_feasible_sol[train].keys()):
            value += (obj["coeff"] * max(0, new_feasible_sol[train][op]["start"] - obj["threshold"]) + obj["increment"] * (1 if new_feasible_sol[train][op]["start"] >= obj["threshold"] else 0))

    return value