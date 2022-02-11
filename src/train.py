import sys
import yaml
import logging
from exceptions import ConfigurationException 
from runner import Config, ExperimentRunner
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
    override_if_not_none(args.seed,"seed",config)
    override_if_not_none(args.dataset,"dataset",config)
    override_if_not_none(args.model,"model",config)
    override_if_not_none(args.gpus,"gpus",config)
    override_if_not_none(args.batch_size,"batch_size",config)
    override_if_not_none(args.learning_rate,"learning_rate",config)
    override_if_not_none(args.validation_list,"validation_list",config)
    override_if_not_none(args.epochs,"epochs",config)


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
    runner = ExperimentRunner(
        config=config,
        root=args.experiments,
        resume=args.resume,
        datasets=args.datasets)

    ## begin experiment
    runner.start()

