from LNS_displib_solver import LnsDisplibSolver
from time import time
from random import sample
from logger import Log

import copy

class LnsCoordinator:
    def __init__(self, instance, feasible_sol, time_limit):
        self.feasible_sol = copy.deepcopy(feasible_sol)
        self.objective = calculate_objective_value(instance.objectives, self.feasible_sol)

        self.start_time = time()
        self.has_to_be_finished_at = self.start_time + time_limit

        self.log = Log(self.feasible_sol, self.objective)

        sample_size = 1 # Experimental
        no_improvement_count = 0 # Experimental
        # do 30 seconds for now since we do not know how long a single iteration of the lns will take
        while self.calculate_remaining_time() > 10:
            if no_improvement_count >= 40: # Experimental
                sample_size += 1
                no_improvement_count = 0
                print(f"Increasing Sample Size to {sample_size}")

            choice = sorted(sample(range(len(feasible_sol)), sample_size))

            new_feasible_sol = LnsDisplibSolver(instance, self.feasible_sol, choice, self.calculate_remaining_time() - 2).solve()
            new_objective_value = calculate_objective_value(instance.objectives, new_feasible_sol)

            if new_objective_value < self.objective:
                no_improvement_count = 0 # Experimental
                print(new_objective_value)
                self.objective = new_objective_value
                self.feasible_sol = new_feasible_sol
                self.log.update_solutions(self.feasible_sol, self.objective)
            else:
                no_improvement_count += 1 # Experimental



    def calculate_remaining_time(self):
        return max(0, self.has_to_be_finished_at - time())


def calculate_objective_value(objectives, new_feasible_sol):
    value = 0
    for obj in objectives:
        train = obj["train"]
        op = obj["operation"]

        if op in new_feasible_sol[train].keys():
            value += (obj["coeff"] * max(0, new_feasible_sol[train][op]["start"] - obj["threshold"]) + obj["increment"] * (new_feasible_sol[train][op]["start"] - obj["threshold"] > 0))

    return value