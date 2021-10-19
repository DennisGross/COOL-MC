import argparse
import sys
import os
sys.path.insert(0, '../')
from common.interpreter.data_generator import *
from common.utilities.project import Project
from common.utilities.training import train
from common.safe_gym.safe_gym import SafeGym
import mlflow
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from joblib import dump, load
from six import StringIO

from sklearn.tree import export_graphviz
from IPython.display import Image  
import pydotplus

def get_arguments():
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = argparse.Namespace()
    unknown_args = list()
    arg_parser.add_argument('--project_dir', help='In which folder should we save your projects?', type=str,
                            default='projects')
    arg_parser.add_argument('--project_name', help='What is the name of your project?', type=str,
                            default='Frozen Lake')
    arg_parser.add_argument('--parent_run_id', help='Do you want to continue training of a RL agent? Name the run_id of the last training unit (see mlflow ui).', type=str,
                            default='')
    arg_parser.add_argument('--task', help='Do you want to continue training of a RL agent? Name the run_id of the last training unit (see mlflow ui).', type=str,
                            default='decision_tree_training')
    arg_parser.add_argument('--prism_dir', help='In which folder should we save your projects?', type=str,
                            default='../prism_files')
    arg_parser.add_argument('--prism_file_path', help='In which folder should we save your projects?', type=str,
                            default='frozen_lake_4x4.prism')
    arg_parser.add_argument('--constant_definitions', help='Do you want to continue training of a RL agent? Name the run_id of the last training unit (see mlflow ui).', type=str,
                            default='slippery=0.04,SIZE=4')
    args, unknown_args = arg_parser.parse_known_args(sys.argv)
    return vars(args)


def create_dataset(agent, env):
    data_folder_path = mlflow.get_artifact_uri(artifact_path="data").replace('/file:/','')
    state_data_path = os.path.join(data_folder_path, 'data.csv')
    data_generator = DataGenerator(state_data_path, env.storm_bridge, agent, env)
    data_generator.label_each_row_with_rl_agent_action()
    features = data_generator.features
    return data_generator.df, features


if __name__ == '__main__':
    command_line_arguments = get_arguments()
    task = command_line_arguments['task']
    prism_file_path = os.path.join(command_line_arguments['prism_dir'], command_line_arguments['prism_file_path'])
    env = SafeGym(prism_file_path, command_line_arguments['constant_definitions'],10, 1000, False, '')
    command_line_arguments['task'] = task
    project = Project(command_line_arguments, env.observation_space, env.action_space)
    if task == 'decision_tree_training':
        df, features = create_dataset(project.agent, env)
        X = df[features]
        y = df['action']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
        #  Create Decision Tree classifer object
        clf = DecisionTreeClassifier()

        # Train Decision Tree Classifer
        clf = clf.fit(X_train,y_train)

        #Predict the response for test dataset
        y_pred = clf.predict(X_test)
        print(X_test, y_pred)
        # Model Accuracy
        acc = metrics.accuracy_score(y_test, y_pred)
        print("Accuracy:",acc)

        dot_data = StringIO()
        export_graphviz(clf, out_file=dot_data,  
                        filled=True, rounded=True,
                        special_characters=True,feature_names = features,class_names=sorted(df.action.unique().tolist()))
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
        #graph.write_png(os.path.join('./', 'decision_tree.png'))
        Image(graph.create_png())
        project.log_accuracy(acc)
        project.save()
        #dump(clf, os.path.join(self.project.project_folder_path, 'decision_tree.joblib')) 