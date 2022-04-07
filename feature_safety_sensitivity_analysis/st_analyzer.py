from copy import copy
import numpy as np
import sys
import sys
import numpy
from numpy.linalg import matrix_power
numpy.set_printoptions(threshold=sys.maxsize)
class State:

    def __init__(self, ID, comment, action, connected_state_IDs, init) -> None:
        self.ID = ID
        self.comment = comment
        self.action = action
        self.connected_state_IDs = connected_state_IDs
        self.init = init


    def __str__(self) -> str:
        return str(self.ID) + "(" + str(self.init) + "):" + str(self.connected_state_IDs) + " with action " + self.action + " " + str(self.comment)

    def get_prob_matrix_row(self, max_state_id):
        row = []
        for id in range(0,max_state_id):
            if id in self.connected_state_IDs.keys():
                row.append(self.connected_state_IDs[id])
            else:
                row.append(0)
        return np.array(row,copy=True)

class DTMC:
    
    def __init__(self, drn_file_path: str) -> None:
        self.states = self.__parse_drn(drn_file_path)


    def __parse_drn(self, drn_file_path):
        f = open(drn_file_path,"r")
        lines = f.readlines()
        f.close()
        states = []
        first = True
        for idx, line in enumerate(lines):
            # If Line starts with state -> NEW state
            if line.find("state")==0:
                state_id = int(line.split(" ")[1].strip())
                comment = ''
                action = ''
                connected_state_IDs = {}
                # Find next state
                for i in range(idx+1, len(lines)):
                    if lines[i].find("state")==0:
                        next_state_line_idx = i
                        break
                    elif i == (len(lines)-1):
                        next_state_line_idx = i+1
                        break
                for i in range(idx+1, next_state_line_idx):
                    if lines[i].find("//")==0:
                        comment = lines[i].replace('\t', ' ')
                    elif lines[i].strip().find("action")==0:
                        action = lines[i].strip().split(" ")[1]
                    elif lines[i].find(":")!=-1:
                        connectd_state_id = int(lines[i].strip().split(':')[0].strip())
                        prob = float(lines[i].strip().split(':')[1].strip())
                        connected_state_IDs[connectd_state_id] = prob
                states.append(State(state_id, comment, action, connected_state_IDs, first))
                print(states[-1].__str__())
                first = False
        #states = sorted(states, key=lambda state: state.ID)
        return states

        
    def get_prob_matrix(self):
        rows = []
        for state in self.states:
            rows.append(state.get_prob_matrix_row(len(self.states)))
        return np.array(rows, copy=True)

    def calc_steady_state(self):
        p = self.get_prob_matrix()
        x_init = np.zeros(p.shape[0])
        sp = 1/p.shape[0]
        for i in range(x_init.shape[0]):
            x_init[i] = sp
        return np.dot(x_init,matrix_power(p,1000))


        

    def get_sum_steady_probs_by_state_ids(self, ids):
        #ids = sorted(ids)
        s = 0
        total_sum = 0
        Pi = self.calc_steady_state()
        for idx, state in enumerate(self.states):
            if state.ID in ids:
                s += Pi[state.ID]
        return (1-s/1)


    def get_init_vector(self):
        rows = []
        for state in self.states:
            init_value = int(state.init)
            rows.append(np.array([init_value]))
        return np.array(rows, copy=True)

    def get_states_by_state_variable_assignment(self, assignment):
        states = []
        idizes = []
        #TODO: FIX multiple numbers
        assignment = assignment + " "
        for idx, state in enumerate(self.states):
            if state.comment.find(assignment)!=-1:
                states.append(state)
                idizes.append(state.ID)
        print(idizes)
        return states, idizes



    

if __name__ == "__main__":
    a = DTMC("test.drn")
    np.set_printoptions(threshold=sys.maxsize)
    M = a.get_prob_matrix()
    ids = a.get_states_by_state_variable_assignment("fuel=1")
    #print(a.get_sum_steady_probs_by_state_ids(ids))
    print(a.calc_steady_state())