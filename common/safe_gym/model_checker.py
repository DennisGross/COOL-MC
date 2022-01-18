
import stormpy
import json
import sys
import time
from common.safe_gym.permissive_manager import PermissiveManager
from common.safe_gym.abstraction_manager import AbstractionManager


class ModelChecker():

    def __init__(self, permissive_input, mapper, abstraction_input):
        self.m_permissive_manager = PermissiveManager(permissive_input, mapper)
        self.m_abstraction_manager = AbstractionManager(mapper, abstraction_input)


    def optimal_checking(self, environment, prop):
        '''
        Uses Storm to model check the PRISM environment.
        :environment, it contains the path to the PRISM file and contains the constant defintions
        :prop, property specifications as a string
        :return model checking result
        '''
        constant_definitions = environment.storm_bridge.constant_definitions
        formula_str = prop
        start_time = time.time()
        prism_program = stormpy.parse_prism_program(environment.storm_bridge.path)
        prism_program = stormpy.preprocess_symbolic_input(prism_program, [], constant_definitions)[0].as_prism_program()
        properties = stormpy.parse_properties(formula_str, prism_program)
        options = stormpy.BuilderOptions([p.raw_formula for p in properties])
        options.set_build_state_valuations()
        model = stormpy.build_sparse_model_with_options(prism_program, options)
        model_size = len(model.states)
        start_checking_time = time.time()
        result = stormpy.model_checking(model, properties[0])
        initial_state = model.initial_states[0]
        mdp_reward_result = result.at(initial_state)
        return mdp_reward_result, model_size, (time.time()-start_time), (time.time()-start_checking_time)

    def __get_clean_state_dict(self, state_valuation_json: str, example_json: str) -> dict:
        '''
        Get the clean state dictionary
        :param state_valuation_json: state valuation as json
        :param example_json: example state json
        :return:
        '''
        state_valuation_json = json.loads(str(state_valuation_json))
        state = {}
        # print(state_valuation_json)
        # print(example_json)
        example_json = json.loads(example_json)
        for key in state_valuation_json.keys():
            for _key in example_json.keys():
                if key == _key:
                    state[key] = state_valuation_json[key]
                    
        
        return state

    def __get_numpy_state(self, env, state_dict):
        state = env.storm_bridge.parse_state(json.dumps(state_dict))
        return state

    def __get_action_for_state(self, env, agent, state):
        '''
        Get the action for the current state
        :param env: environment
        :param state_dict: current state
        :param policy: rl policy
        :return: action name
        '''
        action_index = agent.select_action(state, True)
        return env.action_mapper.actions[action_index], state, action_index


    def induced_markov_chain(self, agent, env, constant_definitions, formula_str = 'Rmin=? [LRA]'):
        '''
        Creates a markov chain of an MDP induced by a Policy and analyze the policy
        :param agent: agent
        :param prism_file: prism file with the MDP
        :param constant_definitions: constants
        :param property_specification: property specification
        :return: mdp_reward_result, model_size, total_run_time, model_checking_time
        '''
        env.reset()
        self.m_permissive_manager.action_mapper = env.action_mapper
        self.wrong_choices = 0
        start_time = time.time()
        prism_program = stormpy.parse_prism_program(env.storm_bridge.path)
        suggestions = dict()
        i = 0
        for m in prism_program.modules:
            for c in m.commands:
                if not c.is_labeled:
                    suggestions[c.global_index] = "tau_" + str(i) #str(m.name)
                    i+=1

        prism_program = stormpy.preprocess_symbolic_input(prism_program, [], constant_definitions)[0].as_prism_program()
        
       
        prism_program = prism_program.label_unlabelled_commands(suggestions)

        properties = stormpy.parse_properties(formula_str, prism_program)
        options = stormpy.BuilderOptions([p.raw_formula for p in properties])
        #options = stormpy.BuilderOptions()
        options.set_build_state_valuations()
        options.set_build_choice_labels(True)
        


        def permissive_policy(state_valuation, action_index):
            """
            Whether for the given state and action, the action should be allowed in the model.
            :param state_valuation:
            :param action_index:
            :return: True or False
            """
            simulator.restart(state_valuation)
            available_actions = sorted(simulator.available_actions())
            action_name = prism_program.get_action_name(action_index)
            # conditions on the action
            state = self.__get_clean_state_dict(
                state_valuation.to_json(), env.storm_bridge.state_json_example)
            state = self.__get_numpy_state(env, state)
            # State Abstraction
            if self.m_abstraction_manager.is_active:
                state = self.m_abstraction_manager.preprocess_state(state)
            
            
            # Check if selected action is available.. if not set action to the first available action
            if len(available_actions) == 0:
                return False
            
            cond1 = False
            if self.m_permissive_manager.is_permissive:
                self.m_permissive_manager.manage_actions(state, agent)
                cond1 = self.m_permissive_manager.create_condition(available_actions, action_name)
            else:
                selected_action, collected_state, collected_action = self.__get_action_for_state(env, agent, state)
                if (selected_action in available_actions) == False:
                    selected_action = available_actions[0]
                #print(state, selected_action)
                cond1 = (action_name == selected_action)

            # print(str(state_valuation.to_json()), action_name)#, state, selected_action, cond1)
            return cond1

        simulator = stormpy.simulator.create_simulator(prism_program)
        simulator.set_action_mode(
            stormpy.simulator.SimulatorActionMode.GLOBAL_NAMES)

        constructor = stormpy.make_sparse_model_builder(prism_program, options,
                                                        stormpy.StateValuationFunctionActionMaskDouble(
                                                            permissive_policy))
        model = constructor.build()
        model_size = len(model.states)
        #print(model)
        #print(formula_str)
        properties = stormpy.parse_properties(formula_str, prism_program)
        #print(properties[0])
        model_checking_start_time = time.time()
        result = stormpy.model_checking(model, properties[0])

        initial_state = model.initial_states[0]
        #print('Result for initial state', result.at(initial_state))
        mdp_reward_result = result.at(initial_state)
        # Update StateActionCollector
        stormpy.export_to_drn(model,"test.drn")
        return mdp_reward_result, model_size, (time.time()-start_time), (time.time()-model_checking_start_time)