
import numpy as np


class ConstantDefinitionParser():
    
    @staticmethod
    def constant_range_to_tuple(constant_range):
        t = []
        parts = constant_range.split(';')
        if len(parts) != 3:
            raise ValueError('Range Format is wrong...')
        for i in range(len(parts)):
            try:
                t.append(int(parts[i]))
            except:
                t.append(float(parts[i]))
        return tuple(t)

    @staticmethod
    def get_range_state_variable(s):
        parts = s.split(',')
        for part in parts:
            if part.count('[') == 1:
                sub_parts = part.split('=')
                return sub_parts[0]
        return None




    @staticmethod
    def parse_constant_definition(s):
        all_constant_definitions = []
        state_variable = ConstantDefinitionParser.get_range_state_variable(s)
        start_idx = s.find('[')
        end_idx = s.find(']')
        constant_range = s[start_idx+1:end_idx]
        tup = ConstantDefinitionParser.constant_range_to_tuple(constant_range)
        if tup[2] < tup[0]:
            raise ValueError("End has to be greater than start")
        for i in np.arange(tup[0], tup[2], tup[1]):
            constant_definition = str(s)
            to_replace = constant_definition[start_idx:(end_idx+1)]
            constant_definition = constant_definition.replace(to_replace, str(i))
            all_constant_definitions.append(constant_definition)
        return all_constant_definitions, tup, state_variable