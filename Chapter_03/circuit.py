from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.base import DAG
from graphics import dagviz
from contextlib import redirect_stdout
from util import d_sep_g


def circuit():
    circuit: BayesianNetwork = BayesianNetwork()
    circuit.add_nodes_from(["X", "Y", "Z", "D", "E", "F", "A1", "A2", "A3", "A4"])
    circuit.add_edge("X", "Z")
    circuit.add_edge("Y", "Z")
    circuit.add_edge("A1", "Z")
    circuit.add_edge("X", "D")
    circuit.add_edge("Z", "D")
    circuit.add_edge("A2", "D")
    circuit.add_edge("Z", "E")
    circuit.add_edge("Y", "E")
    circuit.add_edge("A3", "E")
    circuit.add_edge("D", "F")
    circuit.add_edge("A4", "F")
    circuit.add_edge("E", "F")
    return circuit


def circuit_dag():
    model = circuit()
    G = DAG()
    nodes = model.nodes
    edges = model.edges
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G


def circuit_BN():
    model = circuit()
    cpd_X = TabularCPD(variable="X", variable_card=2, values=[[0.5], [0.5]])
    cpd_Y = TabularCPD(variable="Y", variable_card=2, values=[[0.5], [0.5]])
    cpd_A1 = TabularCPD(variable="A1", variable_card=2, values=[[0.95], [0.05]])
    cpd_A2 = TabularCPD(variable="A2", variable_card=2, values=[[0.95], [0.05]])
    cpd_A3 = TabularCPD(variable="A3", variable_card=2, values=[[0.95], [0.05]])
    cpd_A4 = TabularCPD(variable="A4", variable_card=2, values=[[0.95], [0.05]])
    cpd_Z = TabularCPD(
        variable="Z",
        variable_card=2,
        values=[
            [0, 1, 1, 1, 0.5, 0.5, 0.5, 0.5],
            [1, 0, 0, 0, 0.5, 0.5, 0.5, 0.5],
        ],
        evidence=["A1", "X", "Y"],
        evidence_card=[2, 2, 2],
    )
    cpd_D = TabularCPD(
        variable="D",
        variable_card=2,
        values=[
            [0, 1, 1, 1, 0.5, 0.5, 0.5, 0.5],
            [1, 0, 0, 0, 0.5, 0.5, 0.5, 0.5],
        ],
        evidence=["A2", "X", "Z"],
        evidence_card=[2, 2, 2],
    )
    cpd_E = TabularCPD(
        variable="E",
        variable_card=2,
        values=[
            [0, 1, 1, 1, 0.5, 0.5, 0.5, 0.5],
            [1, 0, 0, 0, 0.5, 0.5, 0.5, 0.5],
        ],
        evidence=["A3", "Y", "Z"],
        evidence_card=[2, 2, 2],
    )
    cpd_F = TabularCPD(
        variable="F",
        variable_card=2,
        values=[
            [0, 1, 1, 1, 0.5, 0.5, 0.5, 0.5],
            [1, 0, 0, 0, 0.5, 0.5, 0.5, 0.5],
        ],
        evidence=["A4", "D", "E"],
        evidence_card=[2, 2, 2],
    )
    model.add_cpds(cpd_X, cpd_Y, cpd_Z, cpd_D, cpd_E, cpd_F)
    return model


def viz_dag(G, name):
    dagviz(G.nodes, G.edges, name)


def print_cpds(model):
    cpds = model.get_cpds()
    for cpd in cpds:
        print(cpd)


def get_markov_blanket(model, node):
    return model.get_markov_blanket(node)


def get_ind(model):
    return model.get_independencies()


def min_dsep(dag, start, end):
    return dag.minimal_dseparator(start, end)


def viz_dag_demo(G, name):
    print(f"visualizing the Bayesian network for the circuit diagnosis example")
    viz_dag(G, name)
    print(
        f"wrote two filee [{name}.dot] and [{name}.png] for specifying and visualizing the {name} DAG"
    )
    print(f"==========================\n")


def min_dsep_demo(G, name, start, end):
    sep = min_dsep(G, start, end)
    print(
        f"minimum d-separator set between {start} and {end} in the DAG  is the set {sep}"
    )
    print(f"==========================\n")


def d_separation_algorithm_demo(G, start, end):
    sep = min_dsep(G, start, end)
    # our algorithm uses lists to represent the sep nodes and the start
    # and end nodes are also lists instead of individual modes as in the
    # min_dsep function of pgmpy
    #
    X = [start]
    Y = [end]
    Z = list(sep)
    print(
        f"executing algorithm d-separation to verify\
that {Z} d-separates the sets {X} and {Y} in the circuit DAG"
    )
    res = d_sep_g(G, X, Y, Z)
    print(f"the results is: {res}")
    print(
        "ancestral graph specification and visualization written to files:\n [anc_g.dot,anc_g.png]"
    )
    print(
        "moral graph specification and visualization written to files:\n [mor_g.dot,mor_g.png]"
    )
    print(f"==========================\n")


def dag_dseparations_demo(G, name):
    print(f"finding all d-separations in the {name} DAG")
    with open(name + "_dag_d-separations.txt", "w") as f:
        with redirect_stdout(f):
            print(G.get_independencies())
    f.close()
    print(
        f"wrote all d-separations for the {name} DAG into file [{name}_dag_d-separations.txt]"
    )
    print(f"==========================\n")


def all_independencies_demo(model, name):
    print(f"finding all independencies in the {name} BN")
    with open(name + "_bn_independencies.txt", "w") as f:
        with redirect_stdout(f):
            print(model.get_independencies())
    f.close()
    print(
        f"wrote all independencies for the {name} BN model into file [{name}_bn_independencies.txt]"
    )
    print(f"==========================\n")


def demo_min_dsep(dag, start, end):
    dag = circuit_dag()
    sep = min_dsep(dag, start="E", end="X")
    print(sep)


def demo_get_independencies():
    chain = DAG([("X", "Y"), ("Y", "Z")])
    return chain.get_independencies()


if __name__ == "__main__":
    print(f"creating the directed acyclic graph for the circuit diagnosis example")
    G = circuit_dag()
    print(G)
    print(f"==========================\n")
    print(f"quantifying the Bayesian network for the circuit diagnosis example")
    model = circuit_BN()
    print_cpds(model)
    print(f"==========================\n")
    viz_dag_demo(G, "circuit")
    min_dsep_demo(G, "circuit", "E", "X")
    d_separation_algorithm_demo(G, "E", "X")
    dag_dseparations_demo(G, "circuit")
    all_independencies_demo(model, "circuit")
