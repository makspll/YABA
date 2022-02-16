import sys
import yaml
import logging
from exceptions import ConfigurationException 
from runner import Config, ExperimentRunner, TestingConfig, TrainingConfig
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

    ## convert config to python object (for static typing support)
    

    try:
        if args.eval:
            config = TestingConfig(config)
        else:
            config = TrainingConfig(config)

    except ConfigurationException as E:
        print(E)
        sys.exit(1)
    
    ## setup logger
    logging.basicConfig(level=args.loglevel)
    
    logging.info(f"Successfully loaded configuration:\n {config}")

    ## setup experiment 
    runner = ExperimentRunner(
        config=config,
        root=args.experiments,
        resume=args.resume,
        datasets=args.datasets)

    ## begin experiment

    if args.eval:
        runner.eval()
    else:
        runner.start()

