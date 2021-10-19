from common.interpreter.data_generator import *
import os
import mlflow
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from joblib import dump, load
from six import StringIO

from sklearn.tree import export_graphviz
from IPython.display import Image  
import pydotplus

class Interpreter():

    def __init__(self, env, project):
        self.env = env
        self.project = project

    def create_dataset(self):
        data_folder_path = mlflow.get_artifact_uri(artifact_path="data").replace('/file:/','')
        state_data_path = os.path.join(data_folder_path, 'data.csv')
        data_generator = DataGenerator(state_data_path, self.env.storm_bridge, self.project.agent, self.env)
        data_generator.label_each_row_with_rl_agent_action()
        features = data_generator.features
        return data_generator.df, features

    def train_and_visualize_decision_tree(self):
        df, features = self.create_dataset()
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
        graph.write_png(os.path.join('./', 'decision_tree.png'))
        Image(graph.create_png())
        self.project.log_accuracy(acc)
        mlflow.log_artifact("decision_tree.png", artifact_path="decision_tree_plot")
        self.project.save()