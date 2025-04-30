import random, functools, logging

from itertools import count
from LNS_displib_solver import LnsDisplibSolver
from time import time
from copy import deepcopy


class LnsCoordinator:
    def __init__(self, original_instance, instance, result, resource_appearances, train_to_resources, time_limit, time_passed):
        self.start_time = time()
        self.time_passed = time_passed
        self.has_to_be_finished_at = self.start_time + time_limit - time_passed

        self.original_instance = original_instance
        self.instance = instance
        self.feasible_sol = result["solution"]
        self.objective = calculate_objective_value(self.instance.objectives, self.feasible_sol)

        self.scc_count = result["scc_count"]
        self.resource_appearances = resource_appearances # Not used ATM
        self.train_to_resources = train_to_resources

        self.train_usage = {train: 0 for train in range(len(self.instance.trains))}
        self.unused_turns = {train: 0 for train in range(len(self.instance.trains))}


    def solve(self):
        # Defined strategies for LNS
        strategy_functions =    [lambda s: self.choose_highest_objective(s),
                                  lambda s: self.choose_resource_conflicted_trains(s),
                                  lambda s: self.choose_random_trains(s),
                                  lambda s: self.choose_longest_unused(s)]

        strategy_names =        ["objective",
                                  "resource_conflicted",
                                  "random",
                                  "longest_unused"]

        mode_names =            ["Semi-fixed",
                                  "Choice",
                                  "Deadlocks"]

        time_limit = 20

        mode = 0
        mode_sizes = [min(15, max(int(round(0.1 * len(self.instance.trains))), 1)), 1, 1]

        strategy = 0
        strat_name = strategy_names[strategy]

        iteration_counter = count()
        nums_of_no_improvement = 0

        logging.info(f"Starting LNS... with mode {mode_names[mode]}, <{strat_name}>\n")
        switch_mode_timer = time()
        start_mode_obj = self.objective

        while self.calculate_remaining_time() > 2:
            next(iteration_counter)
            if mode == 0:
                choice = []
                semi_fixed = strategy_functions[strategy](mode_sizes[mode])
                if len(semi_fixed) == 0:
                    logging.info("Empty semi-fixed list, switching to next strategy.")
                    strategy = (strategy + 1) % len(strategy_functions)
                    semi_fixed = strategy_functions[strategy](mode_sizes[mode])
            else:
                choice = strategy_functions[strategy](mode_sizes[mode])
                semi_fixed = []
                if len(choice) == 0:
                    logging.info("Empty choice list, switching to next strategy.")
                    strategy = (strategy + 1) % len(strategy_functions)
                    choice = strategy_functions[strategy](mode_sizes[mode])

            now = time()
            solver = LnsDisplibSolver(self.instance, self.feasible_sol, choice, semi_fixed, deepcopy(self.train_to_resources), min(time_limit, self.calculate_remaining_time() - 1), mode == 2)
            new_feasible_sol = solver.solve()

            for train in range(len(self.instance.trains)):
                if train in choice + semi_fixed:
                    self.unused_turns[train] = 0
                else:
                    self.unused_turns[train] += 1

            new_objective_value = calculate_objective_value(self.instance.objectives, new_feasible_sol)
            at_min, at_sec = divmod(round(time() - self.start_time + self.time_passed), 60)
            if new_objective_value == 0:
                logging.info(f"{at_min} min {at_sec} sec - Found solution with objective {new_objective_value}.")
                self.objective = new_objective_value
                self.feasible_sol = new_feasible_sol
                break

            if new_objective_value < self.objective:
                logging.info(f"{at_min} min {at_sec} sec - Found solution with better objective {new_objective_value} by rescheduling {choice + semi_fixed} with <{strategy_names[strategy]}>")
                self.objective = new_objective_value
                self.feasible_sol = new_feasible_sol
                if start_mode_obj * 0.9 >= self.objective:
                    switch_mode_timer = time()
                    start_mode_obj = self.objective
                    logging.info("Reset switch_mode_timer because of objective-improvement of at least 10%.")

                nums_of_no_improvement = 0
            elif new_objective_value == self.objective:
                logging.info(f"{at_min} min {at_sec} sec - Found solution with same objective {new_objective_value} by rescheduling {choice + semi_fixed} with <{strategy_names[strategy]}>")
                strategy = (strategy + 1) % len(strategy_functions)
                self.objective = new_objective_value
                self.feasible_sol = new_feasible_sol

                nums_of_no_improvement += 1
            else: # Buggy case
                logging.warning(f"{at_min} min {at_sec} sec - Found solution with worse objective after rescheduling {choice + semi_fixed} with <{strategy_names[strategy]}>. Discarding solution.")
                strategy = (strategy + 1) % len(strategy_functions)

                nums_of_no_improvement += 1

            if time() - now < (time_limit * 0.8) and solver.deadlock_constraints_added:
                mode_sizes[mode] = min(mode_sizes[mode] + 1, len(self.instance.trains))
            else:
                mode_sizes[mode] = max(1, mode_sizes[mode] - 1)

            if mode == 0:
                if time() - switch_mode_timer >= 80 or nums_of_no_improvement == 2 * len(strategy_functions):
                    mode = 1
                    strategy = 0
                    nums_of_no_improvement = 0
                    start_mode_obj = self.objective
                    switch_mode_timer = time()
                    logging.info(f"Switch mode to {mode_names[mode]}")
            elif mode == 1:
                if nums_of_no_improvement >= (2 * len(strategy_functions)):
                    mode = 2
                    self.instance = self.original_instance
                    strategy = 0
                    start_mode_obj = self.objective
                    switch_mode_timer = time()
                    logging.info(f"Switch mode to {mode_names[mode]}")
            else:
                if time() - switch_mode_timer >= 80:
                    time_limit += 10
                    start_mode_obj = self.objective
                    switch_mode_timer = time()
                    logging.info(f"Increasing time limit by 10 seconds to {time_limit}s")

        logging.info(f"LNS iteration-count: {iteration_counter}")
        return self.feasible_sol


    def calculate_remaining_time(self):
        return max(0, self.has_to_be_finished_at - time())


    def choose_longest_unused(self, size):
        choice = sorted(self.unused_turns, key=self.unused_turns.get, reverse=True)[:min(len(self.feasible_sol), size)]
        return choice


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


    def choose_highest_objective(self, size):
        objective_per_train = {i: 0 for i in range(len(self.feasible_sol))}

        for obj in self.instance.objectives:
            train = obj["train"]
            op = obj["operation"]

            if op in self.feasible_sol[train].keys():
                val = objective_per_train[train] + obj["coeff"] * max(0, self.feasible_sol[train][op]["start"] - obj["threshold"]) + obj["increment"] * (self.feasible_sol[train][op]["start"] - obj["threshold"] > 0) # Sometimes, it does not save the new result...
                objective_per_train[train] += val

        choice = sorted(objective_per_train, key=objective_per_train.get, reverse=True)[:min(len(self.feasible_sol), size)]
        return choice


def calculate_objective_value(objectives, new_feasible_sol):
    value = 0
    for obj in objectives:
        train = obj["train"]
        op = obj["operation"]

        if op in list(new_feasible_sol[train].keys()):
            value += (obj["coeff"] * max(0, new_feasible_sol[train][op]["start"] - obj["threshold"]) + obj["increment"] * (1 if new_feasible_sol[train][op]["start"] >= obj["threshold"] else 0))

    return value