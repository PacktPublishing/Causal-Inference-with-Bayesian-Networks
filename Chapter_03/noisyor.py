from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import numpy as np
from pgmpy.factors.discrete import JointProbabilityDistribution


def noisyor_bn():
    noisyor: BayesianNetwork = BayesianNetwork()
    noisyor.add_nodes_from(["x1", "x2", "x3", "y"])
    noisyor.add_edge("x1", "y")
    noisyor.add_edge("x2", "y")
    noisyor.add_edge("x3", "y")
    cpd_1 = TabularCPD(variable="x1", variable_card=2, values=[[0.5], [0.5]])
    cpd_2 = TabularCPD(variable="x2", variable_card=2, values=[[0.5], [0.5]])
    cpd_3 = TabularCPD(variable="x3", variable_card=2, values=[[0.5], [0.5]])
    cpd_y = TabularCPD(
        variable="y",
        variable_card=2,
        values=[
            [1, 0.1, 0.2, 0.02, 0.6, 0.06, 0.12, 0.012],
            [0, 0.9, 0.8, 0.98, 0.4, 0.94, 0.88, 0.988],
        ],
        evidence=["x1", "x2", "x3"],
        evidence_card=[2, 2, 2],
    )
    #%%
    noisyor.add_cpds(cpd_1, cpd_2, cpd_3, cpd_y)
    noisyor.check_model()
    return noisyor


def noisyor_JPD():
    p = 0.5 * 0.5 * 0.5
    val = [
        1,
        0.1,
        0.2,
        0.02,
        0.6,
        0.06,
        0.12,
        0.012,
        0,
        0.9,
        0.8,
        0.98,
        0.4,
        0.94,
        0.88,
        0.988,
    ]
    jp = p * np.array(val)
    print(f"joint probability distribution (JPD) P(Y,X1,X2,X3) is:\n{jp}")
    print(f"sum of the JPD array equals \n{jp.sum()}")
    JPD = JointProbabilityDistribution(["y", "x1", "x2", "x3"], [2, 2, 2, 2], jp)
    return JPD


if __name__ == "__main__":
    model = noisyor_bn()
    jpd = noisyor_JPD()
    res = model.is_imap(jpd)
    print(
        f"model is imap for the joint probability distribution [true or false?]:\n{res}"
    )
