import random, functools, logging
from itertools import count

from LNS_displib_solver import LnsDisplibSolver
from time import time
from copy import deepcopy
from logger import TimeLogger

class LnsCoordinator:
    def __init__(self, instance, result, resource_appearances, train_to_resources, time_limit, time_passed):
        self.start_time = time()
        self.time_passed = time_passed
        self.has_to_be_finished_at = self.start_time + time_limit - time_passed

        self.instance = instance
        self.feasible_sol = result["solution"]
        self.objective = calculate_objective_value(self.instance.objectives, self.feasible_sol)

        self.scc_count = result["scc_count"]
        self.resource_appearances = resource_appearances # Not used ATM
        self.train_to_resources = train_to_resources

        self.train_usage = {train: 0 for train in range(len(self.instance.trains))}
        self.unused_turns = {train: 0 for train in range(len(self.instance.trains))}
        self.mode = -1


    def solve(self):
        # Defined strategies for LNS
        strategy_functions = [lambda s: self.choose_strong_overall_delay(s),
                              # lambda s: self.choose_least_used(s),
                              lambda s: self.choose_resource_conflicted_trains(s),
                              lambda s: self.choose_random_trains(s),
                              lambda s: self.choose_longest_unused(s),
                              # lambda s: self.choose_nearest_threshold(s),
                              # lambda s: self.choose_finished_last(s)
                              ]

        strategy_names = ["objective",
                          # "least_used",
                          "resource_conflicted",
                          "random",
                          "longest_unused",
                          # "nearest_threshold",
                          # "finished_last"
                          ]
        mode_names = ["Semi-fixed",
                      "Choice"]

        # Initialise mode, size of neighborhood, strategy
        self.mode = 1 if (self.scc_count / len(self.instance.trains)) > 0.5 else 0
        strategy = 0
        mode_0_size = 1
        mode_1_size = 1
        strat_name = strategy_names[strategy]

        iteration_counter = count()

        logging.info(f"Starting LNS... with mode {mode_names[self.mode]}, <{strat_name}>\n")
        switch_mode_timer = time()
        start_mode_obj = self.objective

        while self.calculate_remaining_time() > 2:
            next(iteration_counter)
            if self.mode == 0:
                choice = strategy_functions[strategy](mode_0_size)
                semi_fixed = []
                for train in choice:
                    self.train_usage[train] += 1
            else:
                choice = []
                semi_fixed = strategy_functions[strategy](mode_1_size)

            now = time()
            new_feasible_sol = LnsDisplibSolver(self.instance, self.feasible_sol, choice, semi_fixed, deepcopy(self.train_to_resources), self.calculate_remaining_time() - 1, 2).solve()
            if time() - now < 10:
                if self.mode == 0:
                    mode_0_size = min(mode_0_size + 1, len(self.instance.trains))
                else:
                    mode_1_size = min(mode_1_size + 1, len(self.instance.trains))
            else:
                if self.mode == 0:
                    mode_0_size = max(1, mode_0_size - 1)
                else:
                    mode_1_size = max(1, mode_1_size - 1)

            for train in range(len(self.instance.trains)):
                if train in choice + semi_fixed:
                    self.train_usage[train] += 1
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
                logging.info(f"{at_min} min {at_sec} sec - Found solution with better objective {new_objective_value} by rescheduling {choice + semi_fixed} with <{strat_name}>")
                self.objective = new_objective_value
                self.feasible_sol = new_feasible_sol
                if start_mode_obj * 0.9 >= self.objective:
                    switch_mode_timer = time()
                    start_mode_obj = self.objective
                    logging.info("Reset switch_mode_timer because of objective-improvement of at least 10%.")
            elif new_objective_value == self.objective:
                logging.info(f"{at_min} min {at_sec} sec - Found solution with same objective {new_objective_value} by rescheduling {choice + semi_fixed} with {strat_name}")
                strategy = strategy = (strategy + 1) % len(strategy_functions)
                strat_name = strategy_names[strategy]
                self.objective = new_objective_value
                self.feasible_sol = new_feasible_sol
            else:
                # Der LNS-Coordinator wird die size vergrößern, wenn der Solver vorzeitig aufgrund von Fehlern abbricht. Das wird hier ausgebadet
                if self.mode == 0:
                    mode_0_size = max(1, mode_0_size - 1)
                else:
                    mode_1_size = max(1, mode_1_size - 1)

                logging.warning(f"{at_min} min {at_sec} sec - Found solution with worse objective after rescheduling {choice + semi_fixed} with {strat_name}. Discarding solution.")
                strategy = strategy = (strategy + 1) % len(strategy_functions)
                strat_name = strategy_names[strategy]

            if time() - switch_mode_timer >= 60:
                self.mode = 1 - self.mode
                start_mode_obj = self.objective
                switch_mode_timer = time()
                logging.info(f"Switch mode to {mode_names[self.mode]}")

        logging.info(f"LNS iteration-count: {iteration_counter}")
        return self.feasible_sol

    def calculate_remaining_time(self):
        return max(0, self.has_to_be_finished_at - time())


    def choose_least_used(self, size):
        choice = sorted(self.train_usage, key=self.train_usage.get)[:min(len(self.feasible_sol), size)]
        return choice


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


    def choose_nearest_threshold(self, size):
        threshold_overshoot_per_train = {i: 2 ** 40 for i in range(len(self.feasible_sol))}

        for obj in self.instance.objectives:
            train = obj["train"]
            op = obj["operation"]

            if self.feasible_sol[train].get(op):
                diff_to_threshold = max(0, self.feasible_sol[train][op]["start"] - obj["threshold"])
                threshold_overshoot_per_train[train] = min(threshold_overshoot_per_train[train], diff_to_threshold)

        choice = sorted(threshold_overshoot_per_train, key=threshold_overshoot_per_train.get, reverse=True)[:min(len(self.feasible_sol), size)]
        return choice


    def choose_finished_last(self, size):
        return sorted([i for i in range(len(self.instance.trains))],
                      key = lambda x: self.feasible_sol[x][len(self.instance.trains[x])-1]["end"],
                      reverse=True)[:min(len(self.feasible_sol), size)]


    def choose_longest_route(self, size):
        # To be implemented
        pass


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
        return choice


def calculate_objective_value(objectives, new_feasible_sol):
    value = 0
    for obj in objectives:
        train = obj["train"]
        op = obj["operation"]

        if op in list(new_feasible_sol[train].keys()):
            value += (obj["coeff"] * max(0, new_feasible_sol[train][op]["start"] - obj["threshold"]) + obj["increment"] * (1 if new_feasible_sol[train][op]["start"] >= obj["threshold"] else 0))

    return value