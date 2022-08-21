import os
import shutil
import datetime



def date_time_string_to_timestemp(date_time_string):
    """
    Convert date time string to timestemp
    """
    return datetime.datetime.strptime(date_time_string, '%Y-%m-%d %H:%M:%S').timestamp()

def delete_folder_recursively(path, ts):
    """"
    Delete folder if it was created before timestemp ts
    """
    if os.path.exists(path):
        if os.path.getmtime(path) >= ts:
            print("Deleted folder: ", path)
            shutil.rmtree(path)
        else:
            print("Folder not deleted: ", path)

def get_sub_directory_paths_of_folder(path):
    """
    Get sub directory paths of folder
    """
    return [os.path.join(path, name) for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

def clean_folder(path, ts):
    for root_folder_path in get_sub_directory_paths_of_folder(path):
        for folder_path in os.listdir(root_folder_path):
            
            full_path = os.path.join(root_folder_path, folder_path)
            
            # Check if full_path is a folder
            if os.path.isdir(full_path):
                delete_folder_recursively(full_path, ts)

ROOT_PATH='../mlruns/'
ts = date_time_string_to_timestemp('2020-12-01 00:00:00')
clean_folder(ROOT_PATH, ts)
