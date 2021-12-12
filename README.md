# COOL-MC
COOL-MC is an interface between Model Checking and Reinforcement Learning.
It extends the OpenAI Gym to support RL training on PRISM environments
and allows ad-hoc verification of the trained RL agents via the Storm model
checker. The general workflow of our approach is as follows (see also Figure 1).
First, we model the RL environment as a formal model in PRISM. Second, we
train our RL agent in the PRISM environment. Third, we verify the trained RL
agent via the Storm model checker. Depending on the model checking outcome,
we retrain the RL agent or deploy it.
