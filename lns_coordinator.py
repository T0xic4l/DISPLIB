import itertools
import random

from LNS_displib_solver import LnsDisplibSolver
from time import time
from random import sample
from logger import Log

import copy

class LnsCoordinator:
    def __init__(self, instance, feasible_sol, time_limit):
        self.instance = instance
        self.feasible_sol = copy.deepcopy(feasible_sol)
        self.objective = calculate_objective_value(self.instance.objectives, self.feasible_sol)

        self.start_time = time()
        self.has_to_be_finished_at = self.start_time + time_limit

        self.log = Log(self.feasible_sol, self.objective)

        self.no_improvement_count = 0
        self.current_strategy = random.randint(0, 2)

        # actually, 5 seconds might be enough
        while self.calculate_remaining_time() > 5:
            choice = self.choose_trains()

            new_feasible_sol = LnsDisplibSolver(self.instance, self.feasible_sol, choice, self.calculate_remaining_time() - 2).solve()
            new_objective_value = calculate_objective_value(self.instance.objectives, new_feasible_sol)

            if new_objective_value < self.objective:
                print(f"Found a better solution with objective {new_objective_value} by rescheduling {choice}")
                self.objective = new_objective_value
                self.feasible_sol = new_feasible_sol
                self.log.update_solutions(self.feasible_sol, self.objective)
                self.no_improvement_count = 0
            else:
                print(f"No better solution found after rescheduling {choice}")
                self.no_improvement_count += 1


    def calculate_remaining_time(self):
        return max(0, self.has_to_be_finished_at - time())

    '''
    Ideensammlung für Zugauswahl:
    
    VORERST CHECK   1. Zu verschiedenen Zeit-Phasen sollen unterschiedlich viele Züge ausgewählt werden
    CHECK           2. Wähle Züge aus, deren Ressourcen-Überlappung maximal ist
    CHECK           3. Wähle Züge aus, deren Objective-Value am höchsten ist
    VORERST CHECK   4. Wähle Züge komplett zufällig (kann um 5. erweitert werden)
    5. Wähle Züge, die schon lange nicht mehr ausgewählt wurden
    6. Führe ein Scoring auf all diesen aus, um Züge zu wählen, die insgesamt die meisten Kriterien erfüllen
    '''
    def choose_trains(self):
        size = self.choose_choice_size()

        if self.no_improvement_count >= 10:
            self.current_strategy = random.randint(0, 2)
            self.no_improvement_count = 0

        match self.current_strategy:
            case 0:
                return sorted(self.choose_resource_conflicted_trains(size))
            case 1:
                return sorted(self.choose_strongly_delayed_trains(size))
            case 2:
                return sorted(random.sample([i for i in range(len(self.instance.trains))], size))


    def choose_choice_size(self):
        current_time = time()
        if current_time - self.start_time <= 60:
            return 1
        elif current_time - self.start_time <= 200:
            return 2
        elif current_time - self.start_time <= 360 and len(self.instance.trains) <= 100:
            return 3
        elif current_time - self.start_time <= 500 and len(self.instance.trains) <= 50:
            return 4
        else:
            return 3

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


    def choose_strongly_delayed_trains(self, size):
        objective_per_train = {i: 0 for i in range(len(self.feasible_sol))}

        for obj in self.instance.objectives:
            train = obj["train"]
            op = obj["operation"]

            if op in self.feasible_sol[train].keys():
                val = objective_per_train[train] + obj["coeff"] * max(0, self.feasible_sol[train][op]["start"] - obj["threshold"]) + obj["increment"] * (self.feasible_sol[train][op]["start"] - obj["threshold"] > 0) # Sometimes, it does not save the new result...
                objective_per_train[train] += val

        choice = sorted(objective_per_train, key=objective_per_train.get, reverse=True)[:size]
        return sorted(objective_per_train, key=objective_per_train.get, reverse=True)[:min(len(self.feasible_sol), size)] # error handling (size may be invalid)


def calculate_objective_value(objectives, new_feasible_sol):
    value = 0
    for obj in objectives:
        train = obj["train"]
        op = obj["operation"]

        if op in new_feasible_sol[train].keys():
            value += (obj["coeff"] * max(0, new_feasible_sol[train][op]["start"] - obj["threshold"]) + obj["increment"] * (new_feasible_sol[train][op]["start"] - obj["threshold"] >= 0))

    return value