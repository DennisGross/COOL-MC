
class Agent():

    def __init__(self, use_tf_environment = False):
        self.use_tf_environment = use_tf_environment

    def select_action(self, time_step, deploy=False):
        pass

    def store_experience(self, state, action, reward, next_state, terminal):
        pass

    def step_learn(self):
        pass

    def episodic_learn(self):
        pass
    
    def get_hyperparameters(self):
        pass

    def save(self):
        pass

    def load(self, root_folder):
        pass
