# warning: There are some assumptions about the given instance
# 1. There is no release_time for any resource and for any operation
# 2. There is no start_ub for any operation except the first of each train
import copy

from numpy.matlib import empty


def get_heuristic_solution(instance):
    # Increase all zero release_times by one
    for train in instance.trains:
        for i, op in enumerate(train):
            for res in op["resources"]:
                if res["release_time"] == 0 and not succ_uses_res(train, i, res["resource"]):
                    res["release_time"] = 1

    feasible_sol = []

    # Choose a path for each train. All paths will most likely violate several resource constraints
    for i, train in enumerate(instance.trains):
        train_solution = {}
        op = 0
        start_op = 0
        while op != len(train) - 1:
            succ = choose_successor(train, op)
            end_op = max(train[succ]["start_lb"], start_op + train[op]["min_duration"])

            train_solution.update({op: {"start": start_op, "end": end_op, "resources": instance.trains[i][op]["resources"]}})

            op = succ
            start_op = end_op

        train_solution.update({op: {"start": start_op, "end": start_op + train[op]["min_duration"], "resources": instance.trains[i][op]["resources"]}})

        feasible_sol.append(train_solution)

    end_of_last_op = get_end_of_last_op(feasible_sol)
    current_time = 0
    blocked_resources = [] # track resources that are blocked at current time or longer; (r1, freed_time = end+release_time)
    train_schedule = [list(train.keys()) for train in feasible_sol]
    while current_time <= end_of_last_op:
        # update resource_tracker because resources might be freed now
        update_resource_tracker(blocked_resources, current_time)

        for i, train in enumerate(feasible_sol):

            if not len(train_schedule[i]):
                continue # This train is done

            current_operation = train_schedule[i].pop(0)
            print(current_operation)

            if current_time == train[current_operation]["start"]:
                # train wants to start operation, but might be shiftet due to resource conflict
                print("operation startet")
                needed_resources = [res["resource"] for res in train[current_operation]["resources"]]

                if set(needed_resources).isdisjoint(set([res[0] for res in blocked_resources])):
                    # Train gets the resource
                    for res in train[current_operation]["resources"]:
                        blocked_resources.append((res["resource"], train[current_operation]["end"] + res["release_time"]))
                else:
                    # Conflict, train needs to be shiftet
                    min_shift = 0
                    for res in needed_resources:
                        for blocked_res, free_time in blocked_resources:
                            if res == blocked_res:
                                if free_time > min_shift:
                                    min_shift = free_time

                    shift_operations(train, min_shift)

        current_time += 1
        end_of_last_op = get_end_of_last_op(feasible_sol) # This might change due to shift all operations because of resource conflict
        # print(end_of_last_op - current_time)

    return feasible_sol

def choose_successor(train, op):
    find_succ = train[op]["successors"][0]  # initial its the first successor, but there might be a better one
    end_succ = train[find_succ]["start_lb"] + train[find_succ]["min_duration"]

    if len(train[op]["successors"]) > 1:
        for succ in train[op]["successors"]:
            if train[succ]["start_lb"] + train[succ]["min_duration"] < end_succ:
                find_succ = succ
                end_succ = train[succ]["start_lb"] + train[succ]["min_duration"]

    return find_succ


def update_resource_tracker(resources, time):
    resources_iteration = copy.deepcopy(resources)
    for res, free_at in resources_iteration:
        if time == free_at:
            resources.remove((res, free_at))


def shift_operations(train, shift):
    for op, timings in train.items():
        timings["start"] += shift
        timings["end"] += shift


def get_end_of_last_op(feasible_sol):
    find_end = 0
    for train in feasible_sol:
        for op, timings in train.items():
            if timings["end"] > find_end:
                find_end = timings["end"]
    return find_end



def succ_uses_res(train, op, resource):
    for succ in train[op]["successors"]:
        if resource in [res["resource"] for res in train[succ]["resources"]]:
            return True
    return False


'''
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
'''