import os
import shutil
ROOT_PATH='../mlruns/2'



for folder_path in os.listdir(ROOT_PATH):
    task_file = os.path.join(ROOT_PATH, folder_path, "tags/task")

    if os.path.isfile(task_file)==True:
        f = open(task_file)
        content = f.read()
        f.close()
        if content.strip() == "rl_model_checking":
            folder = os.path.join(ROOT_PATH, folder_path)
            if os.path.isdir(folder):
                shutil.rmtree(folder)
