# COOL-MC
COOL-MC provides a variety of environments for reinforcement learning (RL). It is an interface between Model Checking and Reinforcement Learning.
In particular, it extends the OpenAI Gym to support RL training on PRISM environments and allows verification of the trained RL policies via the Storm model checker.
The general workflow of our approach is as follows. First, we model the RL environment as a MDP in PRISM. Second, we train our RL policy in the PRISM environment or, if available, in the matching OpenAI Gym environment. Third, we verify the trained RL policy via the Storm model checker. Depending on the model checking outcome, we retrain the RL policy or deploy it.

![workflow](https://github.com/DennisGross/COOL-MC/blob/main/documentation/images/workflow_diagram.png)


## Start
We assume that you have docker installed and that you run the following commands in the root of this repository:
1. Download the docker container [here](https://drive.google.com/file/d/10C3PkC6uU0M-FEY58zeVOER8CK9zUO3L/view?usp=sharing).
2. Load the docker container: `docker load --input coolmc.tar`
3. Create a project folder: `mkdir projects`
4. Run the docker container: `docker run --user "$(id -u):$(id -g)" -v "$(pwd)/projects":/projects -v "$(pwd)/prism_files":/prism_files -it coolmc bash`

We discuss how to create the docker container yourself, and how to install the tool natively later.

## Example 1 (Frozen Lake)
To demonstrate our tool, we are going to train a RL policy for the OpenAI Gym Frozen Lake environment, and verify it.
The goal is to get familar with our tool and being able to use all our supported features.
