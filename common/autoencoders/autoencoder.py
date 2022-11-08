import torch
from torch.utils.data import Dataset
from os import walk
import os
import numpy as np
import mlflow
import shutil
from common.adversarial_attacks.fgsm import FGSM
import random
import getpass

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



class AEDataLoader(Dataset):

    def __init__(self, root_dir, coop_agent, agent_idx):
        self.root_dir = root_dir
        self.coop_agent = coop_agent
        self.agent_idx = agent_idx
        self.file_paths = []
        for (dirpath, dirnames, filenames) in walk(root_dir):
            for filename in filenames:
                if filename.endswith(".npy"):
                    self.file_paths.append(os.path.join(dirpath, filename))
        self.create_adv_data()

    def create_adv_data(self):
        self.X = []
        self.y = []
        for idx, file_path in enumerate(self.file_paths):
            # Read npy
            x = np.load(file_path)
            x = self.coop_agent.po_manager.get_observation(x, self.agent_idx)
            # numpy to pytorch tensor
            #x = torch.from_numpy(x)
            # Create adversarial data
            rnd_epsilon = random.random()
            m_fgsm = FGSM(self.coop_agent.po_manager.state_mapper, "fgsm,"+str(rnd_epsilon))
            adv_data = m_fgsm.attack(self.coop_agent.agents[self.agent_idx], x)
            adv_data = torch.from_numpy(adv_data)
            #print(adv_data)
            self.X.append(adv_data)
            self.y.append(x)
            self.X.append(x)
            self.y.append(x)
            if idx > 100:
                break
        

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return (self.X[index], self.y[index])


class AE(torch.nn.Module):

    def __init__(self, input_output_size):
        super().__init__()
        self.input_output_size = input_output_size
         
        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        # 784 ==> 9
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_output_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32)
        )
         
        # Building an linear decoder with Linear
        # layer followed by Relu activation function
        # The Sigmoid activation function
        # outputs the value between 0 and 1
        # 9 ==> 784
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, input_output_size),
            torch.nn.Sigmoid()
        )
        self.epsilon = None
        self.to(device)
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def attack_and_clean(self, x, rl_agent, agent_idx):
        # Get first observation of agent
        obs = np.array(rl_agent.po_manager.get_observation(x, agent_idx), copy=True)
        if self.epsilon == None:
            #print(obs)
            obs = torch.from_numpy(obs).to(device)
            #print("Tensor", obs)
            clean_obs = self.forward(obs.float())
            #print("Cleaned", clean_obs)
            x = rl_agent.po_manager.inject_observation_into_state(x, clean_obs, agent_idx)
            #print("Injected", x)
            #exit(0)
            return x
        else:
            m_fgsm = FGSM(rl_agent.po_manager.state_mapper, "fgsm,"+str(self.epsilon))
            #print(obs)
            adv_data = m_fgsm.attack(rl_agent.agents[agent_idx], obs)
            #print("adv_data", adv_data)
            adv_data = torch.from_numpy(adv_data).to(device)
            clean_obs = self.forward(adv_data)
            #print("Cleaned", clean_obs)
            x = rl_agent.po_manager.inject_observation_into_state(x, clean_obs, agent_idx)
            #print("Injected", x)
            return x

    
    def set_attack(self, attack:str):
        self.attack_name, epsilon = attack.split(",")
        self.epsilon = float(epsilon)
        

    def save(self, idx):
        try:
            os.mkdir('tmp_model')
        except Exception as msg:
            username = getpass.getuser()
        torch.save(self.encoder.state_dict(), 'tmp_model/encoder_'+str(self.input_output_size)+'_.chkpt')
        torch.save(self.decoder.state_dict(), 'tmp_model/decoder.chkpt')
        mlflow.log_artifacts("tmp_model", artifact_path="autoencoder" + str(idx))
        shutil.rmtree('tmp_model')


    def load(self, model_path, idx):
        # replace last foldername with autoencoder of the model_path
        model_path = model_path[0].replace("model", "autoencoder" + str(idx))
        self.encoder.load_state_dict(torch.load(os.path.join(model_path, 'encoder_'+str(self.input_output_size)+'_.chkpt')))
        self.decoder.load_state_dict(torch.load(os.path.join(model_path, 'decoder.chkpt')))