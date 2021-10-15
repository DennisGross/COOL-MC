import random

class ActionMapper:
    '''
    The ActionMapper assigns each available action a unique action index [0,...]
    '''

    def __init__(self):
        '''
        Initialize ActionMapper
        '''
        self.actions = []

    def add_action(self, action):
        '''
        Add action if it does not exist
        :param action: action name
        :return:
        '''
        if action not in self.actions:
            self.actions.append(action)
            self.actions.sort()

    def action_index_to_action_name(self, nn_action_idx):
        '''
        Action index (neural network output) to action name
        :param nn_action_idx: Action index (neural network output)
        :return: action name
        '''
        return self.actions[nn_action_idx]

    def action_name_to_action_index(self, action_name):
        '''
        Action name to action index
        :param nn_action_idx: Action index (neural network output)
        :return: action name
        '''
        for i in range(len(self.actions)):
            if action_name == self.actions[i]:
                return i
        return None

    @staticmethod
    def collect_actions(storm_bridge):
        '''
        Collect all actions
        :param action_mapper, store actions
        :storm_bridge to access storm
        :return action_mapper
        '''
        action_mapper = ActionMapper()
        for epoch in range(50):
            storm_bridge.simulator.restart()
            for i in range(1000):
                actions = storm_bridge.simulator.available_actions()
                for action in actions:
                    # Add action if it is not in the list
                    action_mapper.add_action(str(action))
                # Choose randomly an action
                if storm_bridge.simulator.is_done():
                    break
                action_idx = random.randint(0, storm_bridge.simulator.nr_available_actions() - 1)
                storm_bridge.simulator.step(actions[action_idx])
        storm_bridge.simulator.restart()
        return action_mapper