import argparse
import logging

PARSER = argparse.ArgumentParser()

PARSER.add_argument("--config",
    required=True,
    help="Path to configuration file for experiment containing default arguments")

PARSER.add_argument("--experiments",
    default="experiments",
    help="Path to experiments directory")

PARSER.add_argument("--datasets",
    default="datasets",
    help="Path to experiments directory")

PARSER.add_argument("--resume",
    help="Path to experiments directory",
    action="store_const",
    const=True,
    default=False)

PARSER.add_argument("--verbose",
    "-v",
    help="If given will make logging more verbose",
    action="store_const",
    dest="loglevel",
    const=logging.DEBUG,
    default=logging.INFO)

PARSER.add_argument("--eval",
    help="If given loads an evaluation configuration file and only evaluates the model",
    action="store_const",
    dest="eval",
    const=True,
    default=False)

GRAPH_PARSER = argparse.ArgumentParser()

GRAPH_PARSER.add_argument("--experiment_name",
    help="Experiment to graph")

GRAPH_PARSER.add_argument("--graph_type",
    help="the type of graph to produce",
    choices=['gradient_magnitude','gradient_flow'])

GRAPH_PARSER.add_argument("--experiments",
    default="experiments",
    help="Path to experiments directory")

GRAPH_PARSER.add_argument("--verbose",
    "-v",
    help="If given will make logging more verbose",
    action="store_const",
    dest="loglevel",
    const=logging.DEBUG,
    default=logging.INFO)

GRAPH_PARSER.add_argument("--out",
    help="path to save the graph to")

GRAPH_PARSER.add_argument("--show",
    help="show graph as well as save it",
    const=True,
    default=False,
    action="store_const")