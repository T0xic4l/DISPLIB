import json
import os
import time

from event_sorter import EventSorter
from copy import deepcopy

class Log:
    def __init__(self, objectives):
        self.__obj = objectives
        self.__feasible_sol = []

        self.start = time.time()

        self.heuristic_calculation_time = 0
        self.heuristic_rescheduling_counter = 0
        self.lns_iteration_calculation_times = list()
        self.lns_weights = dict()


    def set_solution(self, feasible_sol):
        self.__feasible_sol = deepcopy(feasible_sol)


    def get_solution(self):
        return self.__feasible_sol


    def write_final_solution_to_file(self, path, filename):
        event_sorter = EventSorter(self.__feasible_sol)
        with open(os.path.join(path, filename), 'w') as file:
            file.write(json.dumps({"objective_value": self.calculate_objective_value(), "events": event_sorter.events}))


    def write_log_to_file(self, path, filename):
        with open(os.path.join(path, filename), 'w') as file:
            file.write(json.dumps({"Elapsed time for calculating heuristic solution:" : self.heuristic_calculation_time,
                                   "\nReschedulings needed for heuristic: " : self.heuristic_rescheduling_counter}))
            if self.lns_iteration_calculation_times:
                file.write(json.dumps({"Average elapsed LNS-iteration time: " : sum(self.lns_iteration_calculation_times)/len(self.lns_iteration_calculation_times),
                                       }))

    def calculate_objective_value(self):
        value = 0
        for obj in self.__obj:
            train = obj["train"]
            op = obj["operation"]

            if op in list(self.__feasible_sol[train].keys()):
                value += (obj["coeff"] * max(0, self.__feasible_sol[train][op]["start"] - obj["threshold"]) + obj[
                    "increment"] * (1 if self.__feasible_sol[train][op]["start"] >= obj["threshold"] else 0))

        return value
