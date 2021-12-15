import mlflow
from common.tasks.helper import *


if __name__ == '__main__':
    args = get_arguments()
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