import matplotlib.pyplot as plt
import networkx as nx
import json
import os
from tqdm import tqdm

from data import Instance, Solution
from event_sorter import EventSorter


class Log:
    def __init__(self, start_sol, start_obj_val):
        self.start_sol = start_sol
        self.start_obj_val = start_obj_val

        self.improved_val_sol = []

        self.final_sol = []
        self.final_objective_value = -1


    def update_solutions(self, feasible_sol, obj_val):
        self.improved_val_sol.append((obj_val, feasible_sol))
        self.final_sol = feasible_sol
        self.final_objective_value = obj_val


    def write_final_solution_to_file(self, path, filename):
        event_sorter = EventSorter(self.final_sol)
        with open(os.path.join(path, filename), 'w') as file:
            file.write(json.dumps({"objective_value": self.final_objective_value, "events": event_sorter.events}))


    '''
    def save_res_graph_as_image(self, path, filename):
        pos = nx.spring_layout(self.resource_allocation_graph, seed=42)

        cycles = list(nx.simple_cycles(self.resource_allocation_graph, 2))
        cycle_edges = [edge for cycle in cycles for edge in zip(cycle, cycle[1:] + [cycle[0]])]

        plt.figure(figsize=(40, 40))
        nx.draw(
            self.resource_allocation_graph,
            pos,
            with_labels=True,
            node_size=500,
            font_size=10,
            arrowsize=10,
            node_color="lightgray",
            font_color="black",
            edge_color="lightgray",
            arrows=True
        )

        nx.draw_networkx_edges(
            self.resource_allocation_graph,
            pos,
            edgelist=cycle_edges,
            edge_color="red",
            width=3.0
        )

        plt.savefig(os.path.join(path, filename), format="png")
        plt.close()


    def save_train_graphs_as_image(self, path, filename_prefix):
        for i, graph in enumerate(tqdm(self.train_graphs, desc="Creating Graphs")):
            depth = nx.single_source_shortest_path_length(graph, 0)

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
            plt.figure(figsize=(x, 5))
            nx.draw(
                graph,
                pos,
                with_labels=True,
                node_size=500,
                font_size=10,
                arrowsize=10,
                node_color="lightgray",
                font_color="black",
                edge_color=(200 / 255, 0, 0),
                arrows=True
            )

            plt.savefig(os.path.join(path, f"{filename_prefix}_{i}"), format="png")
            plt.close()
    
    '''

