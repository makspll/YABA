import sys
from typing import Dict, Union
from args import PARSER
import yaml 
import logging
from exceptions import ConfigurationException 
from runner import Config, ExperimentRunner



def err_if_none_arg(arg,key):
    """if arg is none quits program and shows error message

    Args:
        arg ([type]): the value of key or none
        key ([type]): the key name 
    """
    if not arg:
        print(f"{key}: argument missing, needs to be specified in config or argument")
        sys.exit(1)

def override_if_not_none(arg : Union[None,object], key, config : Dict):
    """Overrides key in the config if arg is not None, if no key in config and arg is none
       quits the program with error message    

    Args:
        arg ([type]): the value or none
        key ([type]): the key name 
        config ([type]): config in dictionary form
    """
    if arg:
        config[key] = arg
    else:
        err_if_none_arg(config.get(key,None),key)
        

if __name__ == "__main__":
    args = PARSER.parse_args()


    ## load yaml config
    config = None
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    err_if_none_arg(config,"config")

    ## override config settings with configurable argument names passed
    override_if_not_none(args.experiment_name,"experiment_name",config)
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
    logger = logging.basicConfig(level=args.loglevel)

    ## setup experiment 
    runner = ExperimentRunner(
        config=config,
        root=args.experiments,
        resume=args.resume,
        datasets=args.datasets)

    ## begin experiment
    runner.start()

