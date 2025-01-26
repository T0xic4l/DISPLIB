from LNS_displib_solver import LnsDisplibSolver
from event_sorter import EventSorter
from time import time
from random import sample
from logger import Log

import copy

class LnsCoordinator:
    def __init__(self, instance, feasible_sol, time_limit):
        self.feasible_sol = feasible_sol
        self.objective = calculate_objective_value(instance.objectives, self.feasible_sol)
        self.start_time = time()
        self.has_to_be_finished_at = self.start_time + time_limit

        self.log = Log(copy.deepcopy(self.feasible_sol), self.objective)

        # do 30 seconds for now since we do not know how long a single iteration of the lns will take
        while self.calculate_remaining_time() > 30:
            choice = sorted(sample(range(len(feasible_sol)), 2))

            copy_sol = copy.deepcopy(self.feasible_sol)
            copy_instance = copy.deepcopy(instance)

            new_feasible_sol = LnsDisplibSolver(copy_instance, copy_sol, choice, self.calculate_remaining_time() - 2).solve()
            if new_feasible_sol:
                new_objective_value = calculate_objective_value(instance.objectives, new_feasible_sol)

                if new_objective_value < self.objective:
                    print("improved objective value")
                    self.objective = new_objective_value
                    self.feasible_sol = new_feasible_sol
                    self.log.update_solutions(self.feasible_sol, self.objective)


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