import pygraphviz as pgv


def dagviz(nodes, edges, name):
    G = pgv.AGraph(strict=False, directed=True)
    G.add_nodes_from(nodes)
    for edge in edges:
        G.add_edge(edge)
    G.write(name + ".dot")
    # use dot
    G.layout(prog="dot")
    # write previously positioned graph to PNG file
    G.draw(name + ".png")


def gviz(nodes, edges, name):
    A = pgv.AGraph()
    A.add_nodes_from(nodes)
    for edge in edges:
        A.add_edge(edge)
    A.write(name + ".dot")
    # use dot
    A.layout(prog="dot")
    # write previously positioned graph to PNG file
    A.draw(name + ".png")