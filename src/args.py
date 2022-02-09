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

PARSER.add_argument("--experiment_name",
    help="Overrides 'experiment_name' from the config file provided")

PARSER.add_argument("--dataset",
    help="Overrides 'dataset' from the config file provided")

PARSER.add_argument("--model",
    help="Overrides 'model' from the config file provided")

PARSER.add_argument("--gpus",
    help="Overrides 'gpus' from the config file provided",
    nargs="+",
    type=int)

PARSER.add_argument("--batch_size",
    help="Overrides 'batch_size' from the config file provided",
    type=int)

PARSER.add_argument("--learning_rate",
    help="Overrides 'learning_rate' from the config file provided",
    type=float)

PARSER.add_argument("--validation_list",
    help="Overrides 'validation_list' from the config file provided",
    nargs="+",
    type=int)

PARSER.add_argument("--epochs",
    help="Overrides 'epochs' from the config file provided",
    type=int)