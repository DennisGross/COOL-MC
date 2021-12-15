
import numpy as np
class Agent():

    def __init__(self):
        pass

    def select_action(self, state : np.ndarray, deploy : bool =False):
        """
        The agent gets the OpenAI Gym state and makes a decision.

        Args:
            state (np.ndarray): The state of the OpenAI Gym.
            deploy (bool, optional): If True, do not add randomness to the decision making (e.g. deploy=True in model checking). Defaults to False.
        """
        pass

    def store_experience(self, state  : np.ndarray, action : int, reward : float, next_state : np.ndarray, terminal : bool):
        """
        Stores RL agent training experience.

        Args:
            state (np.ndarray): State
            action (int): Chosen Action
            reward (float): Received Reward
            next_state (np.ndarray): Next State
            terminal (bool): Episode ended?
        """
        pass

    def step_learn(self):
        """
        This method is called every step in the training environment.
        In this method, the agent learns.
        """
        pass

    def episodic_learn(self):
        """
        This method is called in the end of every training episode.
        In this method, the agent learns.
        """
        pass
    
    def get_hyperparameters(self):
        """
        Get the RL agent hyperparameters
        """
        pass

    def save(self):
        """
        Saves the RL agent in the MLFlow experiment.
        """
        pass

    def load(self, root_folder:str):
        """
        Loads the RL agent from the folder

        Args:
            root_folder ([str]): Path the the agent folder
        """
        pass

def to_tuple(number_of_elements : int, values : int):
    '''
    Creates a tuple with
    :param number_of_elements [int]: number of elements
    :param values [int]: element values
    :return: tuple[int]
    '''
    n_tuple: List[int] = []
    for i in range(number_of_elements):
        n_tuple.append(values)
    return tuple(n_tuple)