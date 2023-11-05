from graphics import gviz, dagviz
from pgmpy.base import DAG
from pgmpy.utils import get_example_model
from pgmpy.models import BayesianNetwork


def create_bn(nodes, edges):
    model = BayesianNetwork()
    model.add_nodes_from(nodes)
    model.add_edges_from(edges)
    return model


def d_sep(model, X, Y, Z):
    G = DAG()
    G.add_nodes_from(model.nodes)
    G.add_edges_from(model.edges)
    return d_sep_g(G, X, Y, Z)


def d_sep_g(G, X, Y, Z):
    """
    :param G: directed acyclic graph object of class pgmpy.base.DAG
    :param X: list of nodes
    :param Y: list of nodes disjoint with X
    :param Z: list of separtors disjoint with X and Y
    :return: True if Z d-separates X and Y in G
    """
    # ancestor graph
    anc_dag = G.get_ancestral_graph(nodes=X + Y + Z)
    dagviz(anc_dag.nodes(), anc_dag.edges(), "anc_g")
    # print(anc_dag.edges())
    # moralize
    gmoral = anc_dag.moralize()
    gviz(gmoral.nodes(), gmoral.edges(), "mor_g")
    # remove conditioning nodes
    gmoral.remove_nodes_from(Z)
    gviz(gmoral.nodes(), gmoral.edges(), "result_g")
    # check if nodes in X and Y are separated
    return separated(G, X, Y)


def separated(G, X, Y):
    res = True
    edges = G.edges()
    for x in X:
        for y in Y:
            if (x, y) in edges:
                print(" %s -- %s " % (x, y))
                res = False
                break
    return res


def viz_model(model, name):
    dagviz(model.nodes, model.edges, name)


def print_cpds(model):
    cpds = model.get_cpds()
    for cpd in cpds:
        print(cpd)


def get_markov_blanket(model, node):
    return model.get_markov_blanket(node)


def get_ind(model):
    return model.get_independencies()


def demo_get_ind():
    model = get_example_model("asia")
    get_ind(model)


def min_dsep(dag, start, end):
    return dag.minimal_dseparator(start, end)


def demo_d_sep_g():
    dag = DAG([("A", "B"), ("B", "C")])
    sep = d_sep_g(dag, ["A"], ["C"], ["B"])
    print(sep)


def demo_min_dsep():
    dag = DAG([("A", "B"), ("B", "C")])
    sep = min_dsep(dag, start="A", end="C")
    print(sep)


def demo_d_sep():
    asia = get_example_model("asia")
    res = d_sep(asia, ["either"], ["asia"], ["tub", "lung", "dysp"])
    print(res)


def demo_get_independencies():
    chain = DAG([("X", "Y"), ("Y", "Z")])
    return chain.get_independencies()


if __name__ == "__main__":
    demo_min_dsep()
    demo_get_ind()
    demo_d_sep_g()
    demo_d_sep()
    print(demo_get_independencies())
    asia = get_example_model("asia")
    print_cpds(asia)
    viz_model(asia, "asia")
    # res = d_sep(asia, ["either"], ["asia"], ["tub", "lung", "dysp"])
    res = d_sep(asia, ["tub"], ["dysp"], ["either", "bronc"])
    print(res)
    print(asia.get_markov_blanket("either"))
