import mlflow
import sys
import os
import json
import mlflow
import random
import time
import string
from distutils.dir_util import copy_tree
from mlflow.tracking import MlflowClient
sys.path.insert(0, '..')
from rl_agents.agent_builder import AgentBuilder


class Project():

    def __init__(self, command_line_arguments, observation_space, number_of_actions):
        mlflow.set_tracking_uri(command_line_arguments['project_dir'])
        self.command_line_arguments = command_line_arguments
        self.client = MlflowClient()
        self.project_name = command_line_arguments['project_name']
        self.experiment = self.init_experiment(self.project_name)
        self.run = self.create_new_run(self.command_line_arguments['task'], self.command_line_arguments['parent_run_id'])
        self.start()
        self.agent = self.create_agent(self.command_line_arguments, observation_space, number_of_actions)


    def init_experiment(self, project_name):
        try:
            experiment_id = self.client.create_experiment(project_name)
        except:
            experiment_id = self.client.get_experiment_by_name(project_name).experiment_id
        experiment = self.client.get_experiment(experiment_id)
        return experiment

    def __copy_run(self, experiment, run):
        #new_run = self.client.create_run(self.experiment.experiment_id)
        exists = True
        new_run_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=len(run.info.run_id)+10))
        run_path = os.path.join(command_line_arguments['project_dir'], experiment.experiment_id, run.info.run_id)
        new_run_path = os.path.join(command_line_arguments['project_dir'], experiment.experiment_id, new_run_id)
        while exists:
            new_run_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=len(run.info.run_id)+10))
            new_run_path = os.path.join(command_line_arguments['project_dir'], experiment.experiment_id, new_run_id)
            exists = os.path.exists(new_run_path)
        copy_tree(run_path, new_run_path)
        f = open(os.path.join(new_run_path,'meta.yaml'),'r')
        lines = f.readlines()
        f.close()
        lines[0] = lines[0].replace(run.info.run_id, new_run_id)
        lines[6] = lines[6].replace(run.info.run_id, new_run_id)
        lines[7] = lines[7].replace(run.info.run_id, new_run_id)
        lines[7] = lines[7].replace(run.info.run_id, new_run_id)
        lines[11] = 'start_time: ' + str(time.time()*1000).split('.')[0] + '\n'
        f = open(os.path.join(new_run_path,'meta.yaml'),'w')
        f.writelines(lines)
        f.close()
        run = mlflow.get_run(new_run_id)
        return run



    def create_new_run(self, task, run_id=None):
        if run_id == None or run_id == '':
            # Create new Run
            print("Create new run.")
            run = self.client.create_run(self.experiment.experiment_id)
            self.client.set_tag(run.info.run_id, "task", task)
            return run
        else:
            # Choose already existing run
            print("Choose already existing run.")
            run = mlflow.get_run(run_id)
            new_run = self.__copy_run(self.experiment, run)
            return new_run

    def create_agent(self, command_line_arguments, observation_space, number_of_actions):
        agent = None
        try:
            # Get Model from run artifiact path
            print(mlflow.get_artifact_uri(artifact_path="model"))
            model_folder_path = mlflow.get_artifact_uri(artifact_path="model").replace('/file:/','')
            # Build agent with the model and the hyperparameters
            agent = AgentBuilder.build_agent(model_folder_path, command_line_arguments, observation_space, number_of_actions)
        except:
            # If Model was not saved
            agent = AgentBuilder.build_agent(None, command_line_arguments, observation_space, number_of_actions)
        return agent

    def save(self):
        self.agent.save()
        with open("command_line_arguments.json", 'w') as f:
            json.dump(self.command_line_arguments, f, indent=2)
        mlflow.log_artifact("command_line_arguments.json", artifact_path="meta")
        os.remove('command_line_arguments.json')

    def start(self):
        mlflow.start_run(self.run.info.run_id)

    def close(self):
        mlflow.end_run()


command_line_arguments = {'project_dir':'./projects', 'always_action':0, 'project_name':'frozen_lake42', 'task':'training', 'architecture':'dummy_agent', 'parent_run_id':''}
m_project = Project(command_line_arguments, (2,3), 3)
run_id = m_project.run.info.run_id
m_project.save()
m_project.close()


command_line_arguments = {'project_dir':'./projects', 'always_action':0, 'project_name':'frozen_lake42', 'task':'training', 'architecture':'dummy_agent', 'parent_run_id':run_id}
m_project = Project(command_line_arguments, (2,3), 3) 
m_project.save()
m_project.close()