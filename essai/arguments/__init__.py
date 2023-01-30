from data_args import *
from model_args import *
from training_args import *

def build_arguments(config):
    if config.meta.model_argument == "NI":
        model_args = NIModelArguments
    if config.meta.dataset_argument == "NI":
        data_args = NIDataArguments(**config['data'])
    if config.meta.training_argument == "NI":
        training_args = NITrainingArguments(**config['training'])
    return model_args, data_args, training_args