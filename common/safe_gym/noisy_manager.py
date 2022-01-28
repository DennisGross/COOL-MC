import random
class NoisyFeature:

    def __init__(self, feature_name, lower_bound, upper_bound, feature_idx) -> None:
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.idx = feature_idx

    @staticmethod
    def parse_feature(feature_str, mapper):
        feature_name = feature_str.split('=')[0]
        start_idx_lower_bound = int(feature_str.find("["))+1
        end_idx_lower_bound = int(feature_str.find(";"))
        lower_bound = int(feature_str[start_idx_lower_bound:end_idx_lower_bound])
        start_idx_upper_bound = int(feature_str.find(";"))+1
        end_idx_upper_bound = int(feature_str.find("]"))
        upper_bound = int(feature_str[start_idx_upper_bound:end_idx_upper_bound])
        idx = mapper.mapper[feature_name]
        return NoisyFeature(feature_name, lower_bound, upper_bound, idx)



class NoisyManager:

    def __init__(self, mapper, noisy_feature_str) -> None:
        self.features = []
        if len(noisy_feature_str) > 0:
            for feature_part in noisy_feature_str.split(","):
                feature = NoisyFeature.parse_feature(feature_part, mapper)
                self.features.append(feature)
            self.assignment_storage = {}
        self.noisy_feature_str = noisy_feature_str

    def generate_random_feature_assignments_for_state(self, state):
        for feature in self.features:
            random_assignment = random.randint(feature.lower_bound, feature.upper_bound)
            state[feature.idx] = random_assignment
        return state

    def is_active(self):
        return len(self.noisy_feature_str)>0 and len(self.features) > 0


    def get_random_feature_assignments_for_state(self, state):
        state_str = str(state.tolist())
        if state_str in self.assignment_storage.keys():
            return self.assignment_storage[state_str]
        else:
            self.assignment_storage[state_str] = self.generate_random_feature_assignments_for_state(state)
            return self.assignment_storage[state_str]
