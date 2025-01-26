# warning: There are some assumptions about the given instance
# 1. There is no release_time for any resource and for any operation
# 2. There is no start_ub for any operation except the first of each train

def get_heuristic_solution(instance):
    solution = []
    current_time = 0

    for i, train in enumerate(instance.trains):
        train_solution = {}

        op = 0
        train_solution.update({0: {"start": 0, "end": max(train[0]["min_duration"], current_time), "resources": instance.trains[i][op]["resources"]}})
        current_time = train_solution[0]["end"]

        while op != len(train) - 1:
            op = train[op]["successors"][0]
            current_time = max(current_time, train[op]["start_lb"])
            train_solution.update({op: {"start": current_time, "end": current_time + train[op]["min_duration"], "resources": instance.trains[i][op]["resources"]}})
            current_time = train_solution[op]["end"]

        solution.append(train_solution)

    return solution
