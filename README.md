# COOL-MC
COOL-MC provides a variety of environments for reinforcement learning (RL). It is an interface between Model Checking and Reinforcement Learning.
In particular, it extends the OpenAI Gym to support RL training on PRISM environments and allows verification of the trained RL policies via the Storm model checker.
The general workflow of our approach is as follows. First, we model the RL environment as a MDP in PRISM. Second, we train our RL policy in the PRISM environment or, if available, in the matching OpenAI Gym environment. Third, we verify the trained RL policy via the Storm model checker. Depending on the model checking outcome, we retrain the RL policy or deploy it.

![workflow](https://github.com/DennisGross/COOL-MC/blob/main/documentation/images/workflow_diagram.png)


## Start
We assume that you have docker installed and that you run the following commands in the root of this repository:
1. Download the docker container [here](https://drive.google.com/file/d/10C3PkC6uU0M-FEY58zeVOER8CK9zUO3L/view?usp=sharing) (not up to date).
2. Load the docker container: `docker load --input coolmc.tar`
3. Create a project folder: `mkdir projects`
4. Run the docker container: `docker run --user mycoolmc  -v "$(pwd)/prism_files":"/home/mycoolmc/prism_files" -it coolmc bash`

We discuss how to create the docker container yourself, and how to install the tool natively later.

If you are not familiar with PRISM/Storm, here are some references:

- [PRISM Manual](https://www.prismmodelchecker.org/manual/)
- [Storm Getting Started](https://www.stormchecker.org/getting-started.html)

## Example 1 (Frozen Lake)
FrozenLake is a commonly used OpenAI Gym benchmark, where 
the agent has to reach the goal (frisbee) on a frozen lake. The movement direction of the agent is uncertain and only depends in $33.33\%$ of the cases on the chosen direction. In $66.66\%$ of the cases, the movement is noisy.
To demonstrate our tool, we are going to train a RL policy for the OpenAI Gym Frozen Lake environment, verify it, and retrain it in the PRISM environment.
The following command trains the RL policy in the OpenAI Gym FrozenLake-v0 environment.

`python cool_mc.py --task=openai_training --project_name="Frozen Lake Example" --num_episodes=100 --eval_interval=10 --env=FrozenLake-v0 --sliding_window_size=100 --rl_algorithm=dqn_agent --layers=2 --neurons=64 --replay_buffer_size=30000 --epsilon=1 --epsilon_dec=0.9999 --epsilon_min=0.1 --gamma=0.99 --replace=304 --lr=0.001 --batch_size=32`

Project Specific Arguments:
- `task=openai_training` sets the current task to the OpenAI Gym training.
- `project_name=Frozen Lake Example` is the name of YOUR project. Labeling experiments with the same project name allows the comparison between the runs.
- `num_episodes=100` defines the number of training episodes.
- `eval_inteveral=10` defines the evaluation interval of the RL episode.
- `env=FrozenLake-v0` defines the OpenAI Gym.
- `sliding_window=100` is the size of the sliding window for the reward.

Reinforcement Learning Arguments:
- `rl_algorithm=dqn_agent` defines the reinforcement learning algorithm.
- `layers=2` defines the number of neural network layers.
- `neurons=64` defines the number of neurons in each layer.
- `replay_buffer=30000` defines the size of the replay buffer.
- `epsilon=1` defines the starting epsilon value.
- `epsilon_dec=0.9999` defines the epsilon decay (new_epsilon=current_epsilon * epsilon_dec).
- `epsilon_min=0.1` defines the minimal epsilon value.
- `gamma=0.99` defines the gamma value.
- `replace=304` defines the target network replacing interval.
- `lr=0.001` is the learning rate argument.

After the training, we receive an Experiment ID (e.g. 27f2bbe444754ac0bbbce1326a410419).
We need this ID to identify the experiment.
Now it is possible to verify the trained RL policy via the COOL-MC verifier by passing the experiment ID and the model checking arguments:

`python cool_mc.py --project_name "Frozen Lake Example" --parent_run_id=27f2bbe444754ac0bbbce1326a410419 --task rl_model_checking --prism_file_path="frozen_lake3-v1.prism" --constant_definitions="control=0.33,start_position=0" --prop="Pmin=? [F WATER=true]"`

Project Specific Arguments:
- `project_name "Frozen Lake Example"` specifies the project name.
- `parent_run_id=XXXX` reference to the trained RL policy (use experiment ID).
- `task=rl_model_checking` sets the current task to model checking the trained RL policy.

Model Checking Arguments:
- `prism_file_path="frozen_lake3-v1.prism"` specifies the PRISM environment.
- `constant_definitions="control=0.33,start_position=0"` sets the constant definitions for the PRISM environment.
- `prop="Pmin=? [F WATER=true]"` the property query.

It is also possible to plot the property results over a range of PRISM constant definitions. This is useful, when we want to get a overview of the trained RL policy. In the following command, we plot from different frozen lake agent startpositions 0-15 (stepsize 1, 16 excluded) the probability of falling into the water.

`python cool_mc.py --project_name "Frozen Lake Example" --parent_run_id=27f2bbe444754ac0bbbce1326a410419 --task rl_model_checking --prism_file_path="frozen_lake3-v1.prism" --constant_definitions="control=0.33,start_position=[0;1;16]" --prop="Pmin=? [F WATER=true]"`

`--constant_definitions="control=0.33,start_position=[0;1;16]"` calculates the property results from `start_position=0` up to `start_position=15` with a stepsize of `1`.


If we are not satisfied with the property result, we can retrain the RL policy via the OpenAI Gym or the PRISM environment. The following command, retrains the RL policy in the PRISM environment. 

`python cool_mc.py --task=safe_training  --parent_run_id=27f2bbe444754ac0bbbce1326a410419  --reward_flag=1 --project_name="Frozen Lake Example" --num_episodes=100 --eval_interval=10 --sliding_window_size=100 --rl_algorithm=dqn_agent --layers=2 --neurons=64 --replay_buffer_size=30000 --epsilon=1 --epsilon_dec=0.9999 --epsilon_min=0.1 --gamma=0.99 --replace=304 --lr=0.001 --batch_size=32 --prism_file_path="frozen_lake3-v1.prism" --constant_definitions="control=0.33,start_position=0" --prop="Pmin=? [F WATER=true]"`


- `task=safe_training` specifies the safe training task which allows the RL training in the PRISM environment.
- `prop="Pmin=? [F WATER=true]` tracks the agents probability of falling into the water. `Pmin` also specifies that COOL-MC saves only RL agents which lower probabilities of falling into the water.
- `reward_flag=1` uses rewards instead of penalties.

## Example 2 (Taxi with Fuel)
The taxi agent has to pick up passengers and transport them to their destination without running out of fuel. The environment terminates as soon as the taxi agent does the predefined number of jobs. After the job is done, a new guest spawns randomly at one of the predefined locations.

We train a DQN taxi agent in the PRISM environment:

`python cool_mc.py --task=safe_training --project_name="Taxi with Fuel Example" --num_episodes=100 --eval_interval=10 --sliding_window_size=100 --rl_algorithm=dqn_agent --layers=2 --neurons=64 --replay_buffer_size=30000 --epsilon=1 --epsilon_dec=0.9999 --epsilon_min=0.1 --gamma=0.99 --replace=304 --lr=0.001 --batch_size=32 --prism_file_path="transporter.prism" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prop="Pmin=? [F jobs_done=2]"`


- `project_name="Taxi with Fuel Example"` for the new `Taxi with Fuel Example` project.
- `prop="Pmin=? [F jobs_done=2]"` property query for getting the probability to finish 2 jobs with the trained policy.

After the training, we can verify the trained policy:

`python cool_mc.py --parent_run_id=dd790c269b334e4383b580e7c1da9050 --task=rl_model_checking --project_name="Taxi with Fuel Example" --prism_file_path="transporter.prism" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prop="Pmin=? [F OUT_OF_FUEL=true]"`

State abstraction allows model-checking the trained policy on less precise features without changing the environment. To achieve this, a prepossessing step is applied to the current state in the incremental building process to map the state to a more abstract state for the RL policy. We only have to define a state mapping file and link it to COOL-MC via the command line:

`python cool_mc.py --parent_run_id=dd790c269b334e4383b580e7c1da9050 --task=rl_model_checking --project_name="Taxi with Fuel Example" --prism_file_path="transporter.prism" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prop="Pmin=? [F OUT_OF_FUEL=true]" --abstract_features="../taxi_abstraction.json"`

Permissive Model Checking allows the investigation of the worst-/best-case behaviour of the trained policy for certain state variables.

Minimal Probability of running out of fuel for a fuel level between 4 and 10 (Pmin):
`python cool_mc.py --parent_run_id=dd790c269b334e4383b580e7c1da9050 --task=rl_model_checking --project_name="Taxi with Fuel Example" --prism_file_path="transporter.prism" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prop="Pmin=? [F OUT_OF_FUEL=true]" --permissive_input="fuel<>=[4;10]"`

Maximal Probability of running out of fuel for a fuel level between 4 and 10 (Pmin):
`python cool_mc.py --parent_run_id=dd790c269b334e4383b580e7c1da9050 --task=rl_model_checking --project_name="Taxi with Fuel Example" --prism_file_path="transporter.prism" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prop="Pmax=? [F OUT_OF_FUEL=true]" --permissive_input="fuel<>=[4;10]"`


## Example 3 (Avoid)
Collision avoidance is an environment which contains one agent and two moving obstacles in a two dimensional grid world. The environment terminates as soon as a collision between the agent and one obstacle happens. The environment contains a slickness parameter, which defines the probability that the agent stays in the same cell.

We train a DQN taxi agent in the PRISM environment:

`python cool_mc.py --task=safe_training --project_name="Avoid Example" --num_episodes=100 --eval_interval=10 --sliding_window_size=100 --rl_algorithm=dqn_agent --layers=2 --neurons=64 --replay_buffer_size=30000 --epsilon=1 --epsilon_dec=0.9999 --epsilon_min=0.1 --gamma=0.99 --replace=304 --lr=0.001 --batch_size=32 --prism_file_path="avoid.prism" --constant_definitions="xMax=4,yMax=4,slickness=0.0" --prop="Pmin=? [F COLLISION=true]" --reward_flag=1`

After the training, we can verify the trained policy:

`python cool_mc.py --parent_run_id=915fd49f5f9342a5b5f124dddfd3f15f --task=rl_model_checking --project_name="Avoid Example" --prism_file_path="avoid.prism" --constant_definitions="xMax=4,yMax=4,slickness=0.0" --prop="Pmin=? [F COLLISION=true]"`