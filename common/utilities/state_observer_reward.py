import math

class StateObserverReward:

    def __init__(self, feature_index, feature_assignment) -> None:
        self.feature_index = feature_index
        self.feature_assignment = feature_assignment
        self.active = True
        self.rewards = []


    def observe(self, state, reward):
        #print(state, self.feature_index, self.feature_assignment, state[self.feature_index])
        if int(state[self.feature_index]) == int(self.feature_assignment):
            print("DOOOOOOONE!!")
            self.active = False
            
        
    def reset(self):
        self.active = True


    def after_episode(self):
        if self.active:
            self.rewards.append(0)
        else:
            print("REEEEACHEEED!!!!!")
            self.rewards.append(1)

