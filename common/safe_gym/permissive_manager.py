import numpy as np
class PStateVariable:

    def __init__(self, name, lower_bound, upper_bound):
        self.name = name
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.current_assignment = lower_bound
        self.idx = None

    def next(self):
        current_value = self.current_assignment
        self.current_assignment+=1
        if current_value >= self.upper_bound:
            return None
        else:
            return current_value

    def set_idx(self, idx):
        self.idx = idx

    def reset(self):
        self.current_assignment = self.lower_bound


    def __str__(self):
        return self.name + "=["+str(self.lower_bound)+','+str(self.upper_bound)+"]" + " Index: " + str(self.idx)

    @staticmethod
    def parse_state_variables(permissive_input):
        pstate_variables = []
        print("here",permissive_input)
        for state_variable_str in permissive_input.split(";"):
            print(state_variable_str)
            name = state_variable_str.split('=')[0]
            start_interval = state_variable_str.find('[')
            commata = state_variable_str[start_interval:].find(',')
            end_interval = state_variable_str.find(']')
            lower_bound = int(state_variable_str[start_interval+1:start_interval+commata])
            upper_bound = int(state_variable_str[start_interval+commata+1:end_interval])
            pstate_variable = PStateVariable(name, lower_bound, upper_bound)
            pstate_variables.append(pstate_variable)
        return pstate_variables

    @staticmethod
    def generate_all_states(mapper, fix_state, pstate_variables):
        fix_state = np.array(fix_state, copy=True, dtype=np.int32)
        # Assign all the variable indizes
        for i in range(len(pstate_variables)):
            pstate_variables[i].set_idx(mapper[pstate_variables[i].name])
            print(pstate_variables[i])
        all_states = PStateVariable.__generate_all_states(pstate_variables, fix_state, 0)
        print(all_states)
        return all_states

    @staticmethod
    def __generate_all_states(pstate_variables, state, state_var_idx):
        # Generate new state for current state var
        n_state = state
        all_current_states = []
        if state_var_idx>=len(pstate_variables):
            return []
        for i in range(len(pstate_variables)):
            if i == state_var_idx:
                while True:
                    value = pstate_variables[i].next()
                    if value == None:
                        return all_current_states
                    else:
                        n_state = np.array(n_state, copy=True)
                        n_state[pstate_variables[i].idx] = value
                        all_current_states.append(n_state)
                        sub_states = PStateVariable.__generate_all_states(pstate_variables, n_state, state_var_idx+1)
                        all_current_states.extend(sub_states)
        
        

        

class PermissiveManager:

    def __init__(self, permissive_input, state_var_mapper):
        self.current_state = None
        self.state_var_mapper = state_var_mapper.openai_state_variable_positions
        self.is_permissive = (permissive_input != '')
        self.pstate_variables = PStateVariable.parse_state_variables(permissive_input)
        self.permissive_actions = []

    def manage_actions(self, state, agent):
        if self.current_state == None or self.current_state != state:
            self.current_state = state
            # Get all permissive states
            #print('New', self.current_state)
            all_states = PStateVariable.generate_all_states(self.state_var_mapper, self.current_state, self.pstate_variables)
            for state in all_states:
                action = agent.select_action(state, deploy=True)
                self.permissive_actions.append(action)
            # Get all actions for these permissive states
            print(self.permissive_actions)
        return self.permissive_actions

    def create_condition(self, available_actions, action_name):
        cond1 = False
        for selected_action in self.permissive_actions:
            if (selected_action in available_actions) == False:
                selected_action = available_actions[0]
            cond1 |= (action_name == selected_action)
        return cond1