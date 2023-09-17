from graphics import dagviz
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import BeliefPropagation


def construct_monty_model():
    model: BayesianNetwork = BayesianNetwork()
    model.add_nodes_from(["guest", "car", "host"])
    model.add_edge("guest", "host")
    model.add_edge("car", "host")
    # Defining individual CPDs.
    cpd_g = TabularCPD(
        variable="guest", variable_card=3, values=[[0.33333], [0.33333], [0.33333]]
    )
    cpd_c = TabularCPD(
        variable="car", variable_card=3, values=[[0.33333], [0.33333], [0.33333]]
    )
    cpd_h = TabularCPD(
        variable="host",
        variable_card=3,
        # there are nine states for the parents
        # 11, 12, 13,  21, 22,  23,   31, 32, 33
        values=[
            [0, 0, 0, 0, 0.5, 1, 0, 1, 0.5],
            [0.5, 0, 1, 0, 0, 0, 1, 0, 0.5],
            [0.5, 1, 0, 1, 0.5, 0, 0, 0, 0],
        ],
        evidence=["guest", "car"],
        evidence_card=[3, 3],
    )
    model.add_cpds(cpd_g, cpd_c, cpd_h)
    print(cpd_g)
    print(cpd_c)
    print(cpd_h)
    print(model.check_model())
    return model


def viz_model(model, filename):
    dagviz(model.nodes, model.edges, filename)


if __name__ == "__main__":
    model = construct_monty_model()
    viz_model(model, "monty")
    belief_propagation = BeliefPropagation(model)
    res = belief_propagation.query(variables=["car"], evidence={"guest": 0, "host": 1})
    print(res)
