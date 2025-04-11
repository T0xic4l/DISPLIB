import networkx as nx


def check_properties(instance):
    if repeated_resource_usage(instance.trains):
        print("A train used a resource more than once.")
    else:
        print("No train uses any of its resources no more than once independently")


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

    print(len(graph.nodes))
    print(len(graph.edges))
    print(len(list(nx.simple_cycles(graph))))

    return False