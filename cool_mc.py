import mlflow
from common.tasks.helper import *
import numpy as np
import random
import torch

if __name__ == '__main__':
    args = get_arguments()
    set_random_seed(args['seed'])


    if args['task'] == 'safe_training':
        mlflow.run(
            "safe_gym_training",
            use_conda=False,
            parameters=dict(args)
        )
    elif args['task'] == 'openai_training':
        mlflow.run(
            "openai_gym_training",
            parameters=dict(args),
            use_conda=False
        )
    elif args['task'] == 'rl_model_checking':
        mlflow.run(
            "verify_rl_agent",
            parameters=dict(args),
            use_conda=False
        )
    elif args['task'] == 'sensitivity_analysis':
        mlflow.run(
            "sensitivity_analysis",
            parameters=dict(args),
            use_conda=False
        )