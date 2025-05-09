#!/usr/bin/env python

#
# DISPLIB 2025 verification script v0.3
#
"""
This script verifies a solution to a DISPLIB problem instance.
Usage: displib_verify.py [--test | PROBLEMFILE SOLUTIONFILE]
"""

#
# Changelog:
#  * 2024-10-08: Allow parsing the problem without providing a solution, and check for 
#                referencing (indexing) errors in objective components.
#  * 2024-09-06: Additional checks for topological order, unknown keys, and unordered events.
#  * 2024-08-30: First version with parsing, verification, error reporting, and self-tests.
#


#
#
# Imports and util functions.
#

import sys

if sys.version_info[0] < 3 or sys.version_info[1] < 8:
    print("Must be using at least Python 3.8!")
    exit(1)

from collections import defaultdict
import json, os
import unittest
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

INFINITY = sys.maxsize


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"


def warn(msg):
    print(f"{bcolors.WARNING}WARNING: {bcolors.ENDC}{msg}")


#
#
# Data structures for the problem instance.
#


@dataclass
class ResourceUsage:
    resource: str
    release_time: int


@dataclass
class Operation:
    start_lb: int
    start_ub: int
    min_duration: int
    resources: List[ResourceUsage]
    successors: List[int]


@dataclass
class ObjectiveComponent:
    type: Literal["op_delay"]
    train: int
    operation: int
    threshold: int
    increment: int
    coeff: int


@dataclass
class Problem:
    trains: List[List[Operation]]
    objective: List[ObjectiveComponent]


#
#
# Parsing and consistency checking of the problem file.
#


class ProblemParseError(Exception):
    pass


class SolutionParseError(Exception):
    pass


class SolutionValidationError(Exception):
    def __init__(self, message, relevant_event_idxs=None):
        super().__init__(message)
        self.relevant_event_idxs = relevant_event_idxs


def parse_problem(raw_problem) -> Problem:
    if not isinstance(raw_problem, dict):
        raise ProblemParseError("problem must be a JSON object")

    def parse_operation(train_idx: int, op_idx: int, op_json: dict) -> Operation:
        for key in op_json.keys():
            if key not in ["start_lb","start_ub","min_duration","resources","successors"]:
                raise ProblemParseError(f"unknown key '{key}' in train {train_idx} operation {op_idx}")
            
        if (
            not "successors" in op_json
            or not isinstance(op_json["successors"], list)
            or not all(isinstance(s, int) for s in op_json["successors"])
        ):
            raise ProblemParseError(
                f"'successors' key of operation {op_idx} on train {train_idx} must be a list of positive integers"
            )

        # Check that operations are given in topological order
        for next in op_json["successors"]:
            if next <= op_idx:
                raise ProblemParseError(f"train {train_idx}'s operations are not topologically ordered")

        return Operation(
            start_lb=op_json.get("start_lb", 0),
            start_ub=op_json.get("start_ub", INFINITY),
            min_duration=op_json.get("min_duration", 0),
            resources=[
                ResourceUsage(r.get("resource"), r.get("release_time", 0)) for r in op_json.get("resources", [])
            ],
            successors=op_json.get("successors"),
        )

    for key in raw_problem.keys():
        if key not in ["trains","objective"]:
            raise ProblemParseError(f"unknown key in problem '{key}'")

    if (
        "trains" not in raw_problem
        or not isinstance(raw_problem["trains"], list)
        or not all(isinstance(tr, list) for tr in raw_problem["trains"])
        or not all(all(isinstance(op, dict) for op in tr) for tr in raw_problem["trains"])
    ):
        raise ProblemParseError('problem must have "trains" key mapping to a list of lists of objects')
    trains = [
        [parse_operation(train_idx, op_idx, op) for op_idx, op in enumerate(t)]
        for train_idx, t in enumerate(raw_problem["trains"])
    ]

    # Check that entry and exit operations are unique
    for train_idx, train in enumerate(trains):
        entry_ops = set(i for i, _ in enumerate(train)) - set(i for op in train for i in op.successors)
        if len(entry_ops) == 0:
            raise ProblemParseError(f"train {train_idx} has no entry operation")
        if len(entry_ops) >= 2:
            raise ProblemParseError(f"train {train_idx} has multiple entry operations: {entry_ops}")

        exit_ops = [i for i, t in enumerate(train) if len(t.successors) == 0]
        if len(exit_ops) == 0:
            raise ProblemParseError(f"train {train_idx} has no exit operation")
        if len(exit_ops) >= 2:
            raise ProblemParseError(f"train {train_idx} has multiple exit operations: {exit_ops}")

    def parse_objective_component(idx: int, obj) -> ObjectiveComponent:
        for key in obj.keys():
            if key not in ["type","train","operation","coeff","increment","threshold"]:
                raise ProblemParseError(f"unknown key '{key}' in objective component at index {idx}")
            
        # Check that objective components have valid references to train and operation.
        if obj["train"] < 0 or obj["train"] >= len(trains):
            raise ProblemParseError(f"invalid train reference '{obj['train']}' in objective component {idx}")
        if obj["operation"] < 0 or obj["operation"] >= len(trains[obj["train"]]):
            raise ProblemParseError(f"invalid operation reference '{obj['operation']}' for train '{obj['train']}' in objective component {idx}")

        if obj["type"] != "op_delay":
            raise ProblemParseError(f"objective component at index {idx} has unknown type")
        cmp = ObjectiveComponent(
            type=obj["type"],
            train=obj.get("train"),
            operation=obj.get("operation"),
            coeff=obj.get("coeff", 0),
            increment=obj.get("increment", 0),
            threshold=obj.get("threshold", 0),
        )
        if cmp.increment < 0 or cmp.coeff < 0:
            raise ProblemParseError(f"objective component {idx}: coeff and increment must be nonnegative.")
        if cmp.increment == 0 and cmp.coeff == 0:
            warn(f"objective component {idx}: coeff and increment are both zero.")
        return cmp

    if (
        "objective" not in raw_problem
        or not isinstance(raw_problem["objective"], list)
        or not all(isinstance(obj, dict) for obj in raw_problem["objective"])
    ):
        raise ProblemParseError('problem must have "objective" key with a list value')
    objective = [parse_objective_component(i, o) for i, o in enumerate(raw_problem["objective"])]

    return Problem(trains, objective)


#
#
# Data structures for the solution.
#


@dataclass
class Event:
    time: int
    train: int
    operation: int


@dataclass
class Solution:
    objective_value: int
    events: List[Event]


#
#
# Parsing and consistency checking of the solution file.
#


def parse_solution(raw_solution) -> Solution:
    for key in raw_solution.keys():
        if key not in ["objective_value","events"]:
            raise ProblemParseError(f"unknown key '{key}' in solution object")
        
    if not isinstance(raw_solution, dict):
        raise SolutionParseError("solution must be a JSON object")

    if not "objective_value" in raw_solution or not isinstance(raw_solution["objective_value"], int):
        warn("solution contains no objective value.")

    if (
        not "events" in raw_solution
        or not isinstance(raw_solution["events"], list)
        or not all(isinstance(e, dict) for e in raw_solution["events"])
    ):
        raise SolutionParseError(f'solution object must contain "events" key mapping to a list of objects')

    for i, event in enumerate(raw_solution["events"]):
        for key in event.keys():
            if key not in ["train","time","operation"]:
                raise ProblemParseError(f"unknown key '{key}' in solution event {i}")
        if (
            not isinstance(event["time"], int)
            or not isinstance(event["train"], int)
            or not isinstance(event["operation"], int)
        ):
            raise SolutionParseError(
                f'object at "events" index {i} must contain integer values for keys "time", "train", and "operation"'
            )

    events = [Event(event["time"], event["train"], event["operation"]) for event in raw_solution["events"]]
    return Solution(raw_solution.get("objective_value", INFINITY), events)


#
#
# Verification of the solution.
#


def verify_solution(problem: Problem, solution: Solution):

    # Index the operation delay objective components by operation
    op_delays = {(d.train, d.operation): d for d in problem.objective}

    # The previous start event for each train, given as an index in the event list
    # or `None` when no events for the train have occurred yet.
    train_prev_events: List[Optional[int]] = [None for _ in problem.trains]

    @dataclass
    class OccupiedResource:
        start_event_idx: int
        end_time: int
        release_time: int

    # For each resource, a list of previously started operations that use this resource,
    # that could potentially cause conflicts, stored as indices into the event list. 
    # The events in this list are not necessarily still occupying the resource. We remove the 
    # finished operations eventually, when a new operation aquires the resource and the previous 
    # operation can no longer cause conflicts (i.e., it has finished and the release time has passed).
    resources_occupied: Dict[str, List[OccupiedResource]] = defaultdict(list)

    objective_value = 0
    for event_idx, event in enumerate(solution.events):

        if event_idx > 0 and not (event.time >= solution.events[event_idx - 1].time):
            raise SolutionValidationError(
                f"event {event_idx} starts earlier than the previous event",
                relevant_event_idxs=[event_idx - 1, event_idx],
            )

        if event.train < 0 or event.train >= len(problem.trains):
            raise SolutionValidationError(
                f"event {event_idx} refers to invalid train index", relevant_event_idxs=[event_idx]
            )

        train = problem.trains[event.train]

        if event.operation < 0 or event.operation >= len(train):
            raise SolutionValidationError(
                f"event {event_idx} refers to invalid operation index", relevant_event_idxs=[event_idx]
            )

        operation = train[event.operation]
        train_prev_event = train_prev_events[event.train]

        # Add to the objective value
        if (event.train, event.operation) in op_delays:
            op_delay = op_delays[(event.train, event.operation)]
            objective_value += op_delay.coeff * max(0, event.time - op_delay.threshold)
            objective_value += op_delay.increment * (1 if event.time >= op_delay.threshold else 0)

        # Update end times for the train's previous occupations.
        if train_prev_event is not None:
            prev_event: Event = solution.events[train_prev_event]
            for usage in problem.trains[prev_event.train][prev_event.operation].resources:
                for occ in resources_occupied[usage.resource]:
                    if occ.start_event_idx == train_prev_event:
                        occ.end_time = event.time

        # Remove occupations that have finished
        for res_name in (usage.resource for usage in operation.resources):
            resources_occupied[res_name] = [
                occ for occ in resources_occupied[res_name] if not (event.time >= occ.end_time + occ.release_time)
            ]

        # Check operation bounds
        if event.time < operation.start_lb:
            raise SolutionValidationError(
                f"event {event_idx} violates the lower bound of the operation's start time",
                relevant_event_idxs=[event_idx],
            )

        if event.time > operation.start_ub:
            raise SolutionValidationError(
                f"event {event_idx} violates the upper bound of the operation's start time",
                relevant_event_idxs=[event_idx],
            )

        # Check minimum duration
        if train_prev_event is not None:
            prev_event: Event = solution.events[train_prev_event]
            min_duration = problem.trains[prev_event.train][prev_event.operation].min_duration
            if prev_event.time + min_duration > event.time:
                raise SolutionValidationError(
                    f"event {event_idx} finished operation started by event {train_prev_event} before its minimum duration has passed",
                    relevant_event_idxs=[train_prev_event, event_idx],
                )

        # Check that the operation is a successor of its predecessor.
        if train_prev_event is not None:
            prev_event: Event = solution.events[train_prev_event]
            prev_op = problem.trains[prev_event.train][prev_event.operation]
            if not event.operation in prev_op.successors:
                raise SolutionValidationError(
                    f"event {event_idx} starts an operation that is not a successor of the train's previous operation started by event {train_prev_event}",
                    relevant_event_idxs=[train_prev_event, event_idx],
                )

        else:
            if any(event.operation in op.successors for op in train):
                raise SolutionValidationError(
                    f"event {event_idx} is the first event for train {event.train} but the operation {event.operation} is not an entry operation",
                    relevant_event_idxs=[event_idx],
                )

        # Allocate the resources
        for usage in operation.resources:
            # Check for existing allocations
            occs = resources_occupied[usage.resource]
            for occ in occs:
                other_train = solution.events[occ.start_event_idx].train
                if other_train != event.train:
                    raise SolutionValidationError(
                        f"event {event_idx} allocates resource {usage.resource} which is already allocated to train {other_train}",
                        relevant_event_idxs=[occ.start_event_idx, event_idx],
                    )

            occs.append(OccupiedResource(event_idx, INFINITY, usage.release_time))

        train_prev_events[event.train] = event_idx

    # Check that all trains have finished
    for train_idx, last_event_idx in enumerate(train_prev_events):
        if last_event_idx is None:
            raise SolutionValidationError(f"train {train_idx} has no events")
        else:
            last_event = solution.events[last_event_idx]
            op = problem.trains[last_event.train][last_event.operation]
            if len(op.successors) > 0:
                raise SolutionValidationError(
                    f"train {train_idx} did not finish in its exit operation", relevant_event_idxs=[last_event_idx]
                )

    return objective_value


#
#
# Main function for verifying a solution and writing diagnostic information to the standard output.
#


def main(problemfilename, solutionfilename):
    try:
        print(f"{bcolors.HEADER}DISPLIB 2025 solution verification{bcolors.ENDC}")

        with open(problemfilename) as f:
            raw_problem = json.load(f)
        problem = parse_problem(raw_problem)
        print(f"{bcolors.OKGREEN}✓{bcolors.ENDC} - problem parsed successfully ({len(problem.trains)} trains and {len(problem.objective)} objective components).")

        if solutionfilename is None:
            return

        with open(solutionfilename) as f:
            raw_solution = json.load(f)
        solution = parse_solution(raw_solution)

        value = verify_solution(problem, solution)
        print(f"{bcolors.OKGREEN}✓{bcolors.ENDC} - solution is feasible with objective value {value}.")
        if solution.objective_value < INFINITY and value != solution.objective_value:
            warn(
                f"the solution's objective value {solution.objective_value} does not match the computed objective value"
            )

    except json.JSONDecodeError as e:
        print(f"{bcolors.FAIL}JSON parsing error{bcolors.ENDC}")
        print(f"  {e}")
        sys.exit(1)
    except ProblemParseError as e:
        print(f"{bcolors.FAIL}Error parsing problem file{bcolors.ENDC} ({problemfilename})")
        print(f"  {str(e)}")
        sys.exit(1)
    except SolutionParseError as e:
        print(f"{bcolors.FAIL}Error parsing solution file{bcolors.ENDC} ({solutionfilename})")
        print(f"  {str(e)}")
        sys.exit(1)
    except SolutionValidationError as e:
        print(f"{bcolors.FAIL}Error verifying solution{bcolors.ENDC} ({problemfilename} + {solutionfilename})")
        print(f"  {str(e)}")

        #
        # Print a relevant excerpt of the solution events and highlight the events involved in the constraint violation.
        #
        if e.relevant_event_idxs is not None:
            print()
            n_events = len(raw_solution["events"])
            last_idx = None
            ellipsis = f"                       {bcolors.OKBLUE}(...){bcolors.ENDC}"
            for relevant_idx in e.relevant_event_idxs:
                for idx in range(max(0, (last_idx or -1) + 1, relevant_idx - 3), min(n_events, relevant_idx + 3)):
                    if (last_idx is not None and idx > last_idx + 1) or (last_idx is None and idx > 0):
                        print(ellipsis)

                    arrow = f"{bcolors.HEADER}-->{bcolors.ENDC}" if idx in e.relevant_event_idxs else "   "

                    print(f" {arrow} {bcolors.OKBLUE}[\"events\"][{idx}]:{bcolors.ENDC} {raw_solution['events'][idx]}")

                    last_idx = idx

            if last_idx + 1 < len(raw_solution["events"]):
                print(ellipsis)
        sys.exit(1)


#
#
# Tests.
#


class TestSolutions(unittest.TestCase):
    problem_str = """{"trains":
    [[{"start_ub":0,"min_duration":5,"resources":[{"resource":"l"}],"successors":[1,2]},
        {"min_duration":5,"successors":[3],"resources":[{"resource":"r1"}]},
        {"min_duration":5,"successors":[3],"resources":[{"resource":"r2"}]},
        {"min_duration":5,"successors":[]}],
    [{"start_ub":0,"min_duration":5,"resources":[{"resource":"r1"}],"successors":[1]},
        {"min_duration":5,"resources":[{"resource":"l"}],"successors":[2]},
        {"min_duration":5,"successors":[]}]],
    "objective":[{"type":"op_delay","train":1,"operation":2,"coeff":1}]}"""
    problem = parse_problem(json.loads(problem_str))

    def test_correct_solution(self):
        correct_solution = parse_solution(
            {
                "objective_value": 10,
                "events": [
                    {"time": 0, "train": 0, "operation": 0},
                    {"time": 0, "train": 1, "operation": 0},
                    {"time": 5, "train": 0, "operation": 2},
                    {"time": 5, "train": 1, "operation": 1},
                    {"time": 10, "train": 1, "operation": 2},
                    {"time": 10, "train": 0, "operation": 3},
                ],
            }
        )

        self.assertEqual(verify_solution(self.problem, correct_solution), 10)

    def test_invalid_refs(self):
        with self.assertRaises(SolutionValidationError) as cm:
            verify_solution(
                self.problem,
                parse_solution({"objective_value": 0, "events": [{"time": 0, "train": 99, "operation": 0}]}),
            )
        self.assertEqual(str(cm.exception), "event 0 refers to invalid train index")

        with self.assertRaises(SolutionValidationError) as cm:
            verify_solution(
                self.problem,
                parse_solution({"objective_value": 0, "events": [{"time": 0, "train": -1, "operation": 0}]}),
            )
        self.assertEqual(str(cm.exception), "event 0 refers to invalid train index")

        with self.assertRaises(SolutionValidationError) as cm:
            verify_solution(
                self.problem,
                parse_solution({"objective_value": 0, "events": [{"time": 0, "train": 0, "operation": 99}]}),
            )
        self.assertEqual(str(cm.exception), "event 0 refers to invalid operation index")

        with self.assertRaises(SolutionValidationError) as cm:
            verify_solution(
                self.problem,
                parse_solution({"objective_value": 0, "events": [{"time": 0, "train": 0, "operation": -1}]}),
            )
        self.assertEqual(str(cm.exception), "event 0 refers to invalid operation index")

    def test_start_time_bounds(self):
        with self.assertRaises(SolutionValidationError) as cm:
            verify_solution(
                self.problem,
                parse_solution({"objective_value": 0, "events": [{"time": 1, "train": 0, "operation": 0}]}),
            )
        self.assertEqual(str(cm.exception), "event 0 violates the upper bound of the operation's start time")

        with self.assertRaises(SolutionValidationError) as cm:
            # We modify the problem to have a lower bound on the start time of
            # the first train's second operation.
            problem = parse_problem(json.loads(self.problem_str))
            problem.trains[0][1].start_lb = 6
            verify_solution(
                problem,
                parse_solution(
                    {
                        "objective_value": 0,
                        "events": [{"time": 0, "train": 0, "operation": 0}, {"time": 5, "train": 0, "operation": 1}],
                    }
                ),
            )
        self.assertEqual(str(cm.exception), "event 1 violates the lower bound of the operation's start time")

    def test_minimum_duration(self):
        with self.assertRaises(SolutionValidationError) as cm:
            verify_solution(
                self.problem,
                parse_solution(
                    {
                        "objective_value": 0,
                        "events": [{"time": 0, "train": 0, "operation": 0}, {"time": 4, "train": 0, "operation": 1}],
                    }
                ),
            )
        self.assertEqual(
            str(cm.exception), "event 1 finished operation started by event 0 before its minimum duration has passed"
        )

    def test_operator_successor(self):
        with self.assertRaises(SolutionValidationError) as cm:
            verify_solution(
                self.problem,
                parse_solution(
                    {
                        "objective_value": 0,
                        "events": [{"time": 0, "train": 0, "operation": 0}, {"time": 5, "train": 0, "operation": 3}],
                    }
                ),
            )
        self.assertEqual(
            str(cm.exception),
            "event 1 starts an operation that is not a successor of the train's previous operation started by event 0",
        )

    def test_unfinished_operation_resource_conflict(self):
        with self.assertRaises(SolutionValidationError) as cm:
            verify_solution(
                self.problem,
                parse_solution(
                    {
                        "objective_value": 0,
                        "events": [
                            {"time": 0, "train": 0, "operation": 0},
                            {"time": 0, "train": 1, "operation": 0},
                            {"time": 5, "train": 1, "operation": 1},
                        ],
                    }
                ),
            )
        self.assertEqual(str(cm.exception), "event 2 allocates resource l which is already allocated to train 0")

    def test_release_time_resource_conflict(self):
        # Here we modify the problem to add a non-zero release time in the
        # first operation of the first train. Then we show a correct solution
        # where the second train waits two time units after the first train has
        # left the resource, and and a wrong example where it waits for only one
        # time unit.
        problem = parse_problem(json.loads(self.problem_str))
        problem.trains[0][0].resources[0].release_time = 2

        correct_solution = parse_solution(
            {
                "objective_value": 0,
                "events": [
                    {"time": 0, "train": 0, "operation": 0},
                    {"time": 0, "train": 1, "operation": 0},
                    {"time": 5, "train": 0, "operation": 2},
                    {"time": 7, "train": 1, "operation": 1},
                    {"time": 10, "train": 0, "operation": 3},
                    {"time": 12, "train": 1, "operation": 2},
                ],
            }
        )

        self.assertEqual(verify_solution(problem, correct_solution), 12)

        with self.assertRaises(SolutionValidationError) as cm:
            incorrect_solution = parse_solution(
                {
                    "objective_value": 0,
                    "events": [
                        {"time": 0, "train": 0, "operation": 0},
                        {"time": 0, "train": 1, "operation": 0},
                        {"time": 5, "train": 0, "operation": 2},
                        {"time": 6, "train": 1, "operation": 1},
                    ],
                }
            )

            verify_solution(problem, incorrect_solution)

        self.assertEqual(str(cm.exception), "event 3 allocates resource l which is already allocated to train 0")

    def test_start_in_entry_operation(self):
        with self.assertRaises(SolutionValidationError) as cm:
            verify_solution(
                self.problem,
                parse_solution({"objective_value": 0, "events": [{"time": 0, "train": 0, "operation": 1}]}),
            )
        self.assertEqual(
            str(cm.exception), "event 0 is the first event for train 0 but the operation 1 is not an entry operation"
        )

    def test_end_in_exit_operation(self):
        with self.assertRaises(SolutionValidationError) as cm:
            verify_solution(
                self.problem,
                parse_solution({"objective_value": 0, "events": [{"time": 0, "train": 0, "operation": 0}]}),
            )
        self.assertEqual(str(cm.exception), "train 0 did not finish in its exit operation")

    def test_time_increases(self):
        with self.assertRaises(SolutionValidationError) as cm:
            verify_solution(
                self.problem,
                parse_solution({
                    "objective_value": 0,
                    "events": [
                        {"time": 0, "train": 0, "operation": 0},
                        {"time": 0, "train": 1, "operation": 0},
                        {"time": 6, "train": 0, "operation": 2},
                        {"time": 5, "train": 1, "operation": 1},
                    ],
                }),
            )
        self.assertEqual(str(cm.exception), "event 3 starts earlier than the previous event")



#
#
# Command line arguments parsing.
#

if __name__ == "__main__":
    # This enables colored ANSI output on Windows:
    os.system("")

    if len(sys.argv) >= 2 and sys.argv[1] == "--test":
        unittest.main(argv=[sys.argv[0]], verbosity=2)
        sys.exit(0)

    if len(sys.argv) not in [2,3]:
        print(__doc__)
        sys.exit(1)

    problemfilename = sys.argv[1]
    solutionfilename = sys.argv[2] if len(sys.argv) == 3 else None
    main(problemfilename, solutionfilename)
