#todo clean this file
import shutil
import os
from os import listdir
from os.path import isfile, join
import numpy as np

import pickle as pk
import time

def removeDirectory(path, verbose=True):
    if (os.path.isdir(path)):
        if (True):  # input("are you sure you want to remove this directory? (Y / N): " + path) == "Y" ):
            shutil.rmtree(path)
    else:
        if (verbose):
            print("No Directory to be romved")


def makeDirectory(path, verbose=True):
    try:
        os.mkdir(path)
    except OSError:
        if (verbose):
            print("Creation of the directory %s failed" % path)
    else:
        if (verbose):
            print("Successfully created the directory %s " % path)


def resetParentDirectory(path, verbose=False):
    path = '/'.join(path.rstrip("/").split("/")[:-1])
    removeDirectory(path, verbose)
    makeDirectory(path, verbose)


def resetDirectory(path, verbose=False):
    removeDirectory(path, verbose)
    makeDirectory(path, verbose)


def resetandseed(parent_dir = None, dirList =None ):
    resetDirectory(parent_dir)

    for dir in dirList:
        makeDirectory(dir)

def cache_data(data, file_path):
    with open(file_path, 'wb') as file:
        pk.dump(data,file)
    print("buffer saved to cache", file_path)  # todo add it to logger


def fetch_from_cache(file_path):
    with open(file_path, 'rb') as file:
        data = pk.load(file)
    print("fetched from cache", file_path)  # todo add it to logger
    return data


def create_run_hierarchy(child_folders,parent_folder):
    if( len(child_folders) ==0):
        makeDirectory(parent_folder)
        leaf_folder = parent_folder+time.strftime("%d-%m-%Y_%H:%M:%S")+"/"
        makeDirectory(leaf_folder)
        return leaf_folder
    else:
        makeDirectory(parent_folder)
        parent_folder = parent_folder + child_folders[0] + "/"
        return create_run_hierarchy(child_folders[1:],parent_folder)

def create_hierarchy(child_folders,parent_folder):
    if( len(child_folders) ==0):
        makeDirectory(parent_folder)
        leaf_folder = parent_folder+time.strftime("%d-%m-%Y_%H:%M:%S")+"/"
        makeDirectory(leaf_folder)
        return leaf_folder
    else:
        makeDirectory(parent_folder)
        parent_folder = parent_folder + child_folders[0] + "/"
        return create_hierarchy(child_folders[1:],parent_folder)



def create_results_dir(list_of_dirs, reset = False):
    results_dir = "./results/"  #todo add this to config file
    new_folder_name = create_hierarchy(list_of_dirs,parent_folder=results_dir)

    if(reset):
        resetDirectory(results_dir)

    return new_folder_name

def round_state(arr, precision=2):
    return np.array([round(i, precision) for i in arr])

