import argparse
from email.policy import default
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

PARSER.add_argument("--experiment_name",

    help="Overrides 'experiment_name' from the config file provided")

PARSER.add_argument("--gpus",
    help="Overrides 'gpus' from the config file provided",
    nargs="+",
    type=int)



GRAPH_PARSER = argparse.ArgumentParser()

GRAPH_PARSER.add_argument("--experiment_name",
    help="Experiment to graph")

GRAPH_PARSER.add_argument("--graph_type",
    help="the type of graph to produce",
    choices=['gradient_magnitude', 'accuracy', 'loss', 'all'])

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


CALC_PARSER = argparse.ArgumentParser()

CALC_PARSER.add_argument("--config",
    help="the yaml config file to parse looking for dataset object or others",
    required=True)

CALC_PARSER.add_argument("--out",
    help="the path to output file",
    default="calc_out.txt")

CALC_PARSER.add_argument("--mode",
    choices=["per pixel mean"],
    required=True)

CALC_PARSER.add_argument("--datasets",
    default="datasets",
    help="Path to experiments directory",
    required=True)