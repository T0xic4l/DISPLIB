import json, os, itertools, tqdm
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx


def check_properties(instance, instance_name):
    '''
    if repeated_resource_usage(instance.trains):
        print("A train used a resource more than once.")
    else:
        print("No train uses any of its resources no more than once independently")
    '''

    # analyze_solution_and_instance(instance, instance_name)
    # resource_duplicates_per_operation(instance, instance_name)
    # check_for_zero_release_times(instance, instance_name)
    analyze_operations_graphs(instance, instance_name)

def repeated_resource_usage(trains):
    graph = nx.DiGraph()

    for i, train in enumerate(trains):
        for j, op in enumerate(train):
            current_resources = [r["resource"] for r in op["resources"]]
            for res in op["resources"]:
                for succ in op["successors"]:
                    for succ_res in train[succ]["resources"]:
                        edge = (res["resource"], succ_res["resource"])
                        '''
                        Wenn ein Zug einen Wechsel von A nach B hat, sollte die Kante A->B nur gesetzt werden, wenn dieser Zug in diesem Moment A nicht besitzt.
                        
                        Diese kleine Regel löscht teils viele Kanten und dadurch umso mehr Kreise im Deadlock-Graph!
                        Zusätzlich haben wir hiermit auch herausgefunden, dass ein Zug eine Ressource maximal EIN MAL durchgehend besitzt und danach nie wieder!
                        '''
                        if edge[0] == edge[1] or edge in graph.edges or succ_res["resource"] in current_resources or res["release_time"] != 0:
                            continue
                        else:
                            graph.add_nodes_from([res["resource"], succ_res["resource"]])
                            graph.add_edge(edge[0], edge[1])

    if len(list(nx.simple_cycles(graph))):
        return True

    return False


def analyze_solution_and_instance(instance, instance_name):
    try:
        with open(os.path.join("HeuristicSolutions", f"heuristic_sol2_{instance_name}"), 'r') as file:
            feasible_solution = json.load(file)
    except FileNotFoundError:
        print(f"No solution found for instance {instance_name} was not found")
        return

    l = [34, 4, 38, 7, 8, 39, 12, 16, 19, 21, 22, 23, 24, 25]

    available_resources_per_train = defaultdict(set)
    for i, train in enumerate(instance.trains):
        if i in l:
            for op in train:
                if len(op["resources"]):
                    available_resources_per_train[i].update(res["resource"] for res in op["resources"])

    available_res_sets_per_train_and_op = defaultdict(lambda: defaultdict(set))
    for i, train in enumerate(instance.trains):
        if i in l:
            for j, op in enumerate(train):
                if len(op["resources"]):
                    available_res_sets_per_train_and_op[i][j].update(res["resource"] for res in op["resources"])

    chosen_res_sets_per_train_and_op = defaultdict(lambda: defaultdict(set))
    for event in feasible_solution["events"]:
        train = event["train"]
        op = event["operation"]
        if len(instance.trains[train][op]["resources"]) and train in l:
            chosen_res_sets_per_train_and_op[train][op].update(res["resource"] for res in instance.trains[train][op]["resources"])

    unique_available_res_sets_per_train = defaultdict(lambda: defaultdict(set))
    for train, ops in available_res_sets_per_train_and_op.items():
        other_trains = set(available_resources_per_train.keys()) - {train}
        other_resources = set().union(*(available_resources_per_train[t] for t in other_trains))

        for op_id, res_set in ops.items():
            if res_set.isdisjoint(other_resources):
                unique_available_res_sets_per_train[train][op_id] = res_set

    return


def resource_duplicates_per_operation(instance, instance_name):
    B = False
    for i, train in enumerate(instance.trains):
        for j, op in enumerate(train):
            used_resources = [res["resource"] for res in op["resources"]]

            if len(used_resources) != len(set(used_resources)):
                B = True
                print(f"Train: {i} - Operation: {j}")

    if not B:
        print(f"No duplicates found for {instance_name}")


def check_for_zero_release_times(instance, instance_name):
    for i, train in enumerate(instance.trains):
        for j, op in enumerate(train):
            for res in op["resources"]:
                if res["release_time"] == 0:
                    print(f"Train: {i} - Operation: {j}")
                    return
    print(f"No zero-release-times found for {instance_name}")


def analyze_operations_graphs(instance, instance_name):
    os.makedirs(f"Graphs/{instance_name[:len(instance_name) - 5]}", exist_ok=True)
    graphs = []

    for train in instance.trains:
        graph = nx.DiGraph()
        graph.add_nodes_from([i for i, _ in enumerate(train)])
        for i, operation in enumerate(train):
            graph.add_edges_from([(i, v) for v in operation["successors"]])
        graphs.append(graph)

    for i, graph in tqdm.tqdm(enumerate(graphs), desc="Creating Graphs"):
        max_y = 0
        for op in instance.trains[i]:
            max_y = max(max_y, len(op["successors"]))
        depth = nx.single_source_shortest_path_length(graph, 0)

        red_nodes = set()
        for obj in instance.objectives:
            if obj["train"] == i:
                red_nodes.add(obj["operation"])

        depth_groups = {}
        for node, d in depth.items():
            if d not in depth_groups:
                depth_groups[d] = []
            depth_groups[d].append(node)

        pos = {}
        for d, nodes in depth_groups.items():
            count = len(nodes)
            for j, node in enumerate(sorted(nodes)):
                pos[node] = (d, -((count - 1) / 2) + j)

        x = nx.shortest_path_length(graph, 0, len(graph.nodes) - 1)
        plt.figure(figsize=(x, max_y + 2))
        nx.draw(
            graph,
            pos,
            with_labels=True,
            node_size=700,
            node_color=["#B03A2E" if node in red_nodes else "lightblue" for node in graph.nodes()],
            font_size=10,
            font_color="black",
            edge_color="gray",
            arrows=True,
            arrowsize=20
        )

        plt.savefig(f"Graphs/{instance_name[:len(instance_name) - 5]}/train_{i}.png", format="png")
        plt.close()