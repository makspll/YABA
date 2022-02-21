import sys
import yaml
import logging
from exceptions import ConfigurationException 
from runner import Config, StandardRunner
from args import PARSER
from common import err_if_none_arg, override_if_not_none
from common.yaml_addons import get_loader 


if __name__ == "__main__":
    args = PARSER.parse_args()

    ## load yaml config
    config = None
    with open(args.config, 'r') as f:
        config = yaml.load(f,get_loader())

    err_if_none_arg(config,"config")

    ## override config settings with configurable argument names passed
    override_if_not_none(args.experiment_name,"experiment_name",config)
    override_if_not_none(args.gpus,"gpus",config)

    ## convert config to python object (for static typing support)
    try:
        config = Config(config)
    except ConfigurationException as E:
        print(E)
        sys.exit(1)
    
    ## setup logger
    logging.basicConfig(level=args.loglevel)
    
    logging.info(f"Successfully loaded configuration:\n {config}")

    ## setup experiment 
    runner = StandardRunner(
        config=config,
        root=args.experiments,
        resume=args.resume,
        datasets=args.datasets)

    ## begin experiment
    runner.start()

