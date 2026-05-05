from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.factors.discrete import JointProbabilityDistribution
import numpy as np


def bn1():
    m1 = BayesianNetwork()
    m1.add_nodes_from(["X", "Y", "Z"])
    m1.get_independencies()
    cpd_X = TabularCPD(variable="X", variable_card=2, values=[[0.4], [0.6]])
    cpd_Y = TabularCPD(variable="Y", variable_card=2, values=[[0.7], [0.3]])
    cpd_Z = TabularCPD(variable="Z", variable_card=2, values=[[0.1], [0.9]])
    m1.add_cpds(cpd_X, cpd_Y, cpd_Z)
    return m1


def bn2():
    m2 = BayesianNetwork()
    m2.add_nodes_from(["X", "Y", "Z"])
    m2.add_edge("Y", "X")
    m2.add_edge("Y", "Z")
    cpd_Y = TabularCPD(variable="Y", variable_card=2, values=[[0.7], [0.3]])
    cpd_X = TabularCPD(
        variable="X",
        variable_card=2,
        values=[
            [0.4, 0.2],
            [0.6, 0.8],
        ],
        evidence=["Y"],
        evidence_card=[2],
    )
    cpd_Z = TabularCPD(
        variable="Z",
        variable_card=2,
        values=[
            [0.5, 0.4],
            [0.5, 0.6],
        ],
        evidence=["Y"],
        evidence_card=[2],
    )
    # print(cpd_X.is_valid_cpd(), cpd_Y.is_valid_cpd(), cpd_Z.is_valid_cpd())
    m2.add_cpds(cpd_X, cpd_Y, cpd_Z)


def imap_demo1():
    m1 = bn1()
    # m1 = BayesianNetwork()
    # m1.add_nodes_from(["X", "Y", "Z"])
    # m1.get_independencies()
    # cpd_X = TabularCPD(variable="X", variable_card=2, values=[[0.4], [0.6]])
    # cpd_Y = TabularCPD(variable="Y", variable_card=2, values=[[0.7], [0.3]])
    # cpd_Z = TabularCPD(variable="Z", variable_card=2, values=[[0.1], [0.9]])
    # m1.add_cpds(cpd_X, cpd_Y, cpd_Z)
    p1 = [0.4, 0.6]
    p2 = [0.7, 0.3]
    p3 = [0.1, 0.9]
    jpd = []
    for i in p1:
        for j in p2:
            for k in p3:
                jpd.append(i * j * k)
    # print(jpd)
    JPD = JointProbabilityDistribution(["X", "Y", "Z"], [2, 2, 2], jpd)
    check_imap = m1.is_imap(JPD)
    print(f"Bayesian network m1 has \n nodes: {m1.nodes} edges: {m1.edges} ")
    print(
        f"the model represents the following joint probability distribution on {m1.nodes}"
    )
    print(jpd)
    print(f"Conditional independencies are:")
    print(m1.get_independencies())
    print(f"Is m1 an I-map of the joint distribution?\nAnswer: {check_imap}")


def imap_demo2():
    m2 = BayesianNetwork()
    m2.add_nodes_from(["X", "Y", "Z"])
    m2.add_edge("Y", "X")
    m2.add_edge("Y", "Z")
    cpd_Y = TabularCPD(variable="Y", variable_card=2, values=[[0.7], [0.3]])
    cpd_X = TabularCPD(
        variable="X",
        variable_card=2,
        values=[
            [0.4, 0.2],
            [0.6, 0.8],
        ],
        evidence=["Y"],
        evidence_card=[2],
    )
    cpd_Z = TabularCPD(
        variable="Z",
        variable_card=2,
        values=[
            [0.5, 0.4],
            [0.5, 0.6],
        ],
        evidence=["Y"],
        evidence_card=[2],
    )
    # print(cpd_X.is_valid_cpd(), cpd_Y.is_valid_cpd(), cpd_Z.is_valid_cpd())
    m2.add_cpds(cpd_X, cpd_Y, cpd_Z)
    # calculate joint probability distribution
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
    jpd = []
    for Y in [0, 1]:
        for X in [0, 1]:
            for Z in [0, 1]:
                p_yxz = np.multiply(np.multiply(P_Y[Y], P_X_Y[X, Y]), P_Z_Y[Z, Y])
                jpd.append(p_yxz[0])
                # print(Y, X, Z, p_yxz)
    # print(jpd)
    # print(sum(jpd))
    prob = JointProbabilityDistribution(["Y", "X", "Z"], [2, 2, 2], jpd)
    check_imap = m2.is_imap(prob)
    print(f"Bayesian network m1 has \n nodes: {m2.nodes} edges: {m2.edges} ")
    print(
        f"the model represents the following joint probability distribution on {m2.nodes}"
    )
    print(jpd)
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
