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

        # actually, 5 seconds might be enough
        while self.calculate_remaining_time() > 5:
            choice = self.choose_trains()

            new_feasible_sol = LnsDisplibSolver(instance, self.feasible_sol, choice, self.calculate_remaining_time() - 2).solve()
            new_objective_value = calculate_objective_value(instance.objectives, new_feasible_sol)

            if new_objective_value < self.objective:
                print(f"Found a better solution with objective {new_objective_value} by rescheduling {choice}")
                self.objective = new_objective_value
                self.feasible_sol = new_feasible_sol
                self.log.update_solutions(self.feasible_sol, self.objective)


    def calculate_remaining_time(self):
        return max(0, self.has_to_be_finished_at - time())


    def choose_trains(self):
        '''
        Ideensammlung für Zugauswahl
        1. Zu verschiedenen Zeit-Phasen sollen unterschiedlich viele Züge ausgewählt werden
        2. Wähle Züge aus, deren Ressourcen-Überlappung maximal ist
        3. Wähle Züge aus, deren Objective-Value am höchsten ist
        4. Wähle Züge komplett zufällig (kann um 5. erweitert werden)
        5. Wähle Züge, die schon lange nicht mehr ausgewählt wurden
        6. Führe ein Scoring auf all diesen aus, um Züge zu wählen, die insgesamt die meisten Kriterien erfüllen
        '''
        return sorted(sample(range(len(self.feasible_sol)), 2))


def calculate_objective_value(objectives, new_feasible_sol):
    value = 0
    for obj in objectives:
        train = obj["train"]
        op = obj["operation"]

        if op in new_feasible_sol[train].keys():
            value += (obj["coeff"] * max(0, new_feasible_sol[train][op]["start"] - obj["threshold"]) + obj["increment"] * (new_feasible_sol[train][op]["start"] - obj["threshold"] > 0))

    return value