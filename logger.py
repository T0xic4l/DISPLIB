import json
import os

from event_sorter import EventSorter
from copy import deepcopy

class Log:
    def __init__(self, feasible_sol, objectives):
        self.__obj = objectives
        self.__feasible_sol = deepcopy(feasible_sol)


    def set_solution(self, feasible_sol):
        self.__feasible_sol = deepcopy(feasible_sol)


    def write_final_solution_to_file(self, path, filename):
        event_sorter = EventSorter(self.__feasible_sol)
        with open(os.path.join(path, filename), 'w') as file:
            file.write(json.dumps({"objective_value": self.calculate_objective_value(), "events": event_sorter.events}))


    def calculate_objective_value(self):
        value = 0
        for obj in self.__obj:
            train = obj["train"]
            op = obj["operation"]

            if op in list(self.__feasible_sol[train].keys()):
                value += (obj["coeff"] * max(0, self.__feasible_sol[train][op]["start"] - obj["threshold"]) + obj[
                    "increment"] * (1 if self.__feasible_sol[train][op]["start"] >= obj["threshold"] else 0))

        return value
