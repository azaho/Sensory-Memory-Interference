import argparse, json, time, random, datetime
import hashlib, torch, math, pathlib, shutil, sys
import numpy as np
from torch import nn
from task_and_training_template import *

# PARSER END

verbose = True  # print info in console?

hyperparameters.update({
    "random_string": str("AA"),  # human-readable string used for random initialization (for reproducibility)
    "regularization": "L2_weights",  # options: L1, L2, None
    "regularization_lambda": 0,

    "learning_rate": 1e-4,
})
task_parameters.update({
    "task_name": "2DIR1O",
    "input_direction_units": 100,  # how many direction-selective input units?
    "dim_input": 100 + 1,  # plus one input for go cue signal
})
model_parameters.update({
    "model_name": "backpropCTRNN",
    "dim_recurrent": 100,
    "dim_input": 100 + 1,  # plus one input for go cue signal
})
additional_comments += [
    "Networks trained with Adam from a random initialization."
]


task_parameters["delay1_from"] = 10
task_parameters["delay1_to"] = 90
task_parameters["delay2_from"] = 120
task_parameters["delay2_to"] = 160
task_parameters["distractor_probability"] = 1.0


directory = update_directory_name()
update_random_seed()

if __name__ == "__main__":
    # train the network and save weights
    model = Model()

    directory = update_directory_name()

    task = Task()
    result = train_network(model, task, directory)

    save_metadata(directory, task, model, result)
    save_training_data(directory, result)
    model.save_firing_rates(task, "data_npy/" + directory[5:-1] + ".npy")
    save_metadata(directory, task, model, result, path="data_npy/" + directory[5:-1] + ".json")
    save_analysis_notebooks(directory, args)
    output_connectivity_factors(directory, task, model)