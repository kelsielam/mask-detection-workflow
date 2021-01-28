import random
from Pegasus.api import *
import numpy as np

def split_data_filenames(filenames):

    random.shuffle(filenames)
    train, validate, test = np.split(filenames, [int(len(filenames)*0.7), int(len(filenames)*0.8)])
    files_split_dict = {}
    files_split_dict["train"] = train
    files_split_dict["test"] = test
    files_split_dict["validate"] = validate
    
    train_filenames = ["train_" + file for file in train] 
    val_filenames = ["val_" + file for file in validate] 
    test_filenames =  ["test_" + file for file in test] 
    return train_filenames,val_filenames,test_filenames, files_split_dict