from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.factors.discrete import JointProbabilityDistribution
from util import d_sep, viz_model
import numpy as np


def bn1(px, py, pz):
    m1 = BayesianNetwork()
    m1.add_nodes_from(["X", "Y", "Z"])
    m1.get_independencies()
    cpd_X = TabularCPD(variable="X", variable_card=2, values=px)
    cpd_Y = TabularCPD(variable="Y", variable_card=2, values=py)
    cpd_Z = TabularCPD(variable="Z", variable_card=2, values=pz)
    m1.add_cpds(cpd_X, cpd_Y, cpd_Z)
    return m1


def bn2(p_y, p_x_y, p_z_y):
    m2 = BayesianNetwork()
    m2.add_nodes_from(["X", "Y", "Z"])
    m2.add_edge("Y", "X")
    m2.add_edge("Y", "Z")
    cpd_Y = TabularCPD(variable="Y", variable_card=2, values=p_y)
    cpd_X = TabularCPD(
        variable="X",
        variable_card=2,
        values=p_x_y,
        evidence=["Y"],
        evidence_card=[2],
    )
    cpd_Z = TabularCPD(
        variable="Z",
        variable_card=2,
        values=p_z_y,
        evidence=["Y"],
        evidence_card=[2],
    )
    # print(cpd_X.is_valid_cpd(), cpd_Y.is_valid_cpd(), cpd_Z.is_valid_cpd())
    m2.add_cpds(cpd_X, cpd_Y, cpd_Z)
    return m2


def imap_demo1():
    px = np.array([[0.4], [0.6]])
    py = np.array([[0.7], [0.3]])
    pz = np.array([[0.1], [0.9]])
    m1 = bn1(px, py, pz)
    jpd = (
        (np.multiply(pz, (np.multiply(px, py.transpose())).flatten()))
        .transpose()
        .flatten()
    )
    JPD = JointProbabilityDistribution(["X", "Y", "Z"], [2, 2, 2], jpd)
    check_imap = m1.is_imap(JPD)
    print(f"Bayesian network BN1 has \n nodes: {m1.nodes} edges: {m1.edges} ")
    print(f"BN1 joint distribution P(x,y,z) is:\n{jpd}")
    imap = m1.get_independencies()
    print(f"BN1 I-map is:\n{imap}:")
    print(f"Is BN1 an I-map of the joint distribution?\nAnswer: {check_imap}")


def imap_demo2():
    P_Y = np.array([[0.7], [0.3]])
    P_X_Y = np.array(
        [
            [0.4, 0.2],
            [0.6, 0.8],
        ]
    )
    P_Z_Y = np.array(
        [
            [0.5, 0.4],
            [0.5, 0.6],
        ]
    )
    m2 = bn2(P_Y, P_X_Y, P_Z_Y)
    jpd = []
    for Y in [0, 1]:
        for X in [0, 1]:
            for Z in [0, 1]:
                p_yxz = np.multiply(np.multiply(P_Y[Y], P_X_Y[X, Y]), P_Z_Y[Z, Y])
                jpd.append(p_yxz[0])
    prob = JointProbabilityDistribution(["Y", "X", "Z"], [2, 2, 2], jpd)
    check_imap = m2.is_imap(prob)
    print(f"Bayesian network BN2 has \n nodes: {m2.nodes} edges: {m2.edges} ")
    print(f"BN2 joint distribution P(x,y,z) is:\n{jpd}")
    print(f"Conditional independencies are:")
    print(m2.get_independencies())
    print(f"Is m2 an I-map of the joint distribution?\nAnswer: {check_imap}")


def create_bn(nodes, edges):
    model = BayesianNetwork()
    model.add_nodes_from(nodes)
    model.add_edges_from(edges)
    return model


def equiv_demo():
    #  first model
    m1 = create_bn(["x", "y", "z"], [("x", "y"), ("y", "z")])
    set1 = m1.get_independencies()
    #  second model
    m2 = create_bn(["x", "y", "z"], [("y", "x"), ("y", "z")])
    set2 = m2.get_independencies()
    #  third model
    m3 = create_bn(["x", "y", "z"], [("z", "y"), ("y", "x")])
    set3 = m3.get_independencies()
    if set1 == set2:
        print(f"m1 and m2 are I-equivalent:\n [{set1}] == [{set2}]")
    if set1 == set3:
        print(f"m1 and m3 are I-equivalent:\n [{set1}] == [{set3}]")
    if set2 == set3:
        print(f"m2 and m3 are I-equivalent:\n [{set2}] == [{set3}]")


def example_bn(name):
    nodes = ["A", "S", "T", "L", "E", "B", "X", "D"]
    edges = [
        ("A", "T"),
        ("S", "L"),
        ("S", "B"),
        ("T", "E"),
        ("L", "E"),
        ("E", "X"),
        ("E", "D"),
        ("B", "D"),
    ]
    m = create_bn(nodes, edges)
    print(f"created Bayesian network model {name}")
    viz_model(m, name)
    print(f"Nodes:\n{nodes}\nEdges\n{edges}")
    return m


def demo_graphoid_contraction():
    name = "graphoid_demo"
    m = example_bn(name)
    print(f"Nodes:\n{m.nodes}\nEdges\n{m.edges}")
    X = ["B"]
    Y = ["T"]
    Z = ["S"]
    W = ["A", "L", "E", "X"]
    print(f"created graph layout inf files: {name}.dot and {name}.png")
    res1 = d_sep(m, ["B"], ["S"], ["T"])
    print(f"I({X}, {Z}, {Y}) is {res1}")
    #%%
    res2 = d_sep(m, ["B"], ["S", "T"], ["A", "L", "E", "X"])
    print(f"I({X}, {Z} U {Y}, {W}) is {res2}")
    #%%
    res3 = d_sep(m, ["B"], ["S"], ["T", "A", "L", "E", "X"])
    print(f"I({X}, {Z}, {W} U {Y}) is {res3}")
    print(f"verified contraction axiom")


def demo_graphoid_intersection():
    name = "graphoid_demo"
    m = example_bn(name)
    print(f"Nodes:\n{m.nodes}\nEdges\n{m.edges}")
    X = ["E"]
    Y = ["A"]
    W = ["S"]
    Z = ["T", "D", "B", "X", "L"]
    print(f"created graph layout inf files: {name}.dot and {name}.png")
    res1 = d_sep(m, ["E"], ["A"], ["T", "D", "B", "X", "L", "S"])
    print(f"I({X}, {Z} U {W}, {Y}) is {res1}")
    #%%
    res2 = d_sep(m, ["E"], ["S"], ["T", "D", "B", "X", "L", "A"])
    print(f"I({X}, {Z} U {Y}, {W}) is {res2}")
    #%%
    res3 = d_sep(m, ["E"], ["S", "A"], ["T", "D", "B", "X", "L"])
    print(f"I({X}, {Z}, {W} U {Y}) is {res3}")
    print(f"verified intersection axiom")


def markov_blanket_demo():
    name = "mb_demo"
    m = example_bn(name)
    node = "L"
    blanket = m.get_markov_blanket(node)
    print(f"the Markov blanket for node {node} is {blanket}")
    # find the set of nodes outside the blanket
    all = set(m.nodes)
    all.remove(node)
    outside = list(all.difference(set(blanket)))
    print(f"nodes outside the blanket: {outside}")
    # verify the blanket independence condition
    res = d_sep(m, [node], outside, blanket)
    print(f"I([{node}], {blanket}, {outside}) is {res}")
    print(f"verified Markov blanket\n=======================")


if __name__ == "__main__":
    print(f"=======================")
    print(f"I-map demo 1\n=======================")
    imap_demo1()
    print(f"=======================")
    print(f"I-map demo 2\n=======================")
    imap_demo2()
    print(f"=======================")
    print(f"I-equiv demo\n=======================")
    equiv_demo()
    print(f"=======================")
    print(f"Graphoid contraction demo\n=======================")
    demo_graphoid_contraction()
    print(f"=======================")
    print(f"Graphoid intersection demo\n=======================")
    demo_graphoid_intersection()
    print(f"=======================")
    print(f"Markov blanket demo\n=======================")
    markov_blanket_demo()
