from typing import Literal

from langgraph.constants import START, END
from langgraph.graph import StateGraph

from src.node import Nodes
from src.schemas.state import FeatureState, Decision


def route_after_evaluation(
    state: FeatureState,
) -> Literal["generate_features", "improve_features", "save_features"]:
    if state.attempt >= state.max_attempt:
        return "save_features"
    if state.decision == Decision.RETRY:
        return "generate_features"
    if state.decision == Decision.IMPROVE:
        return "improve_features"
    return "save_features"


def build_graph(llm, df_train, df_test, description):
    nodes = Nodes(llm=llm, df_train=df_train, df_test=df_test, description=description)

    builder = StateGraph(FeatureState)

    builder.add_node("generate_features", nodes.generate_features)
    builder.add_node("improve_features", nodes.improve_features)
    builder.add_node("evaluate_features", nodes.evaluate_features)
    builder.add_node("save_features", nodes.save_features)

    builder.add_edge(START, "generate_features")
    builder.add_edge("generate_features", "evaluate_features")
    builder.add_edge("improve_features", "evaluate_features")

    builder.add_conditional_edges(
        "evaluate_features",
        route_after_evaluation,
        {
            "generate_features": "generate_features",
            "improve_features": "improve_features",
            "save_features": "save_features",
        },
    )

    builder.add_edge("save_features", END)
    return builder