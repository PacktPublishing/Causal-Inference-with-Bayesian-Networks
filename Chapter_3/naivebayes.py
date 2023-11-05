import pandas as pd
from pgmpy.models import NaiveBayes
from pgmpy.models import BayesianNetwork
from pgmpy.inference import BeliefPropagation
from pgmpy.factors.discrete import TabularCPD
from graphics import dagviz


weather = NaiveBayes()
values = pd.read_csv("play.csv")

weather.fit(values, "play")
for cpd in weather.get_cpds():
    print(cpd)


def create_classifier():
    classifier: BayesianNetwork = BayesianNetwork()
    classifier.add_nodes_from(["outlook", "temperature", "humidity", "windy", "play"])
    classifier.add_edges_from(
        [
            ("play", "outlook"),
            ("play", "temperature"),
            ("play", "humidity"),
            ("play", "windy"),
        ]
    )
    dagviz(classifier.nodes(), classifier.edges(), "weather")

    cpd_play = TabularCPD(variable="play", variable_card=2, values=[[0.36], [0.64]])
    # print(cpd_play.values)
    # print(cpd_play.variables)
    # print(cpd_play.is_valid_cpd())
    cpd_outlook = TabularCPD(
        variable="outlook",
        variable_card=3,
        values=[
            [0.0, 0.45],
            [0.4, 0.33],
            [0.6, 0.22],
        ],
        evidence=["play"],
        evidence_card=[2],
    )
    # print(cpd_outlook.values)
    # print(cpd_outlook.variables)
    # print(cpd_outlook.is_valid_cpd())
    cpd_temperature = TabularCPD(
        variable="temperature",
        variable_card=3,
        values=[
            [0.2, 0.33],
            [0.4, 0.22],
            [0.4, 0.45],
        ],
        evidence=["play"],
        evidence_card=[2],
    )
    # print(cpd_temperature.values)
    # print(cpd_temperature.variables)
    # print(cpd_temperature.is_valid_cpd())
    cpd_humidity = TabularCPD(
        variable="humidity",
        variable_card=2,
        values=[
            [0.8, 0.33],
            [0.2, 0.67],
        ],
        evidence=["play"],
        evidence_card=[2],
    )
    # print(cpd_humidity.values)
    # print(cpd_humidity.variables)
    # print(cpd_humidity.is_valid_cpd())
    cpd_windy = TabularCPD(
        variable="windy",
        variable_card=2,
        values=[
            [0.4, 0.67],
            [0.6, 0.33],
        ],
        evidence=["play"],
        evidence_card=[2],
    )
    # print(cpd_windy.values)
    # print(cpd_windy.variables)
    # print(cpd_windy.is_valid_cpd())

    classifier.add_cpds(cpd_play, cpd_outlook, cpd_temperature, cpd_humidity, cpd_windy)
    return classifier


states = {
    "play": {"no": 0, "yes": 1},
    "outlook": {"overcast": 0, "rainy": 1, "sunny": 2},
    "temperature": {"cool": 0, "hot": 1, "mild": 2},
    "humidity": {"high": 0, "normal": 1},
    "windy": {"False": 0, "True": 1},
}


def state_dict(var):
    return states[var]


def state_index(var, state):
    dict = state_dict(var)
    return dict[state]


def map_evidence(dict):
    for k in dict:
        # print(k)
        val = dict[k]
        # print(val)
        val2 = state_index(k, val)
        # print(val2)
        dict[k] = val2
    return dict


# print(map_evidence({"outlook": "rainy", "windy": "False"}))

# =============================


def classify(classifier, evidence):
    belief_propagation = BeliefPropagation(classifier)
    evidence2 = map_evidence(evidence)
    res = belief_propagation.query(
        # variables=["play"], evidence={"outlook": 2, "humidity": 0}
        variables=["play"],
        evidence=evidence2,
    )
    return res


if __name__ == "__main__":
    weather = NaiveBayes()
    values = pd.read_csv("play.csv")
    weather.fit(values, "play")
    for cpd in weather.get_cpds():
        print(cpd)
    classifier = create_classifier()
    print(classifier.check_model())
    # If outlook = sunny and humidity = high then play = no
    res1 = classify(classifier, {"outlook": "sunny", "humidity": "high"})
    print(f"If outlook = sunny and humidity = high then\n{res1}")
    # If outlook = overcast and temperature = mild and windy = False then play = no
    res2 = classify(
        classifier, {"outlook": "overcast", "temperature": "mild", "windy": "False"}
    )
    print(
        f"If outlook = overcast and temperature = mild and windy = False then\n{res2}"
    )
