from pgmpy.utils import get_example_model
from util import d_sep, viz_model, print_cpds

# def d_sep(model, X, Y, Z):
#     G = DAG()
#     G.add_nodes_from(model.nodes)
#     G.add_edges_from(model.edges)
#     # ancestor graph
#     anc_dag = G.get_ancestral_graph(nodes=X + Y + Z)
#     dagviz(anc_dag.nodes(), anc_dag.edges(), "anc_g")
#     # print(anc_dag.edges())
#     # moralize
#     gmoral = anc_dag.moralize()
#     gviz(gmoral.nodes(), gmoral.edges(), "mor_g")
#     # remove conditioning nodes
#     gmoral.remove_nodes_from(["tub", "lung", "dysp"])
#     gviz(gmoral.nodes(), gmoral.edges(), "result_g")
#     # check if nodes in X connected to nodes in Y in the moral graph
#     return connected(G, X, Y)


# def get_markov_blanket(model, node):
#     return model.get_markov_blanket(node)


def connected(G, X, Y):
    res = True
    edges = G.edges()
    for x in X:
        for y in Y:
            if (x, y) in edges:
                print(" %s -- %s " % (x, y))
                res = False
                break
            else:
                print("no edge: (%s, %s) " % (x, y))
    return res


def independencies(dag):
    dag.get_independencies()


# def viz_model(model, name):
#     dagviz(model.nodes, model.edges, name)


# def print_cpds(model):
#     cpds = model.get_cpds()
#     for cpd in cpds:
#         print(cpd)


if __name__ == "__main__":
    model = get_example_model("asia")
    print_cpds(model)
    print(model.local_independencies("either"))
    # viz_model(asia, "asia")
    # # print(independencies(asia))
    # res = d_sep(model, ["either"], ["asia"], ["tub", "lung", "dysp"])
    # print(res)
    # print(asia.get_markov_blanket("either"))
