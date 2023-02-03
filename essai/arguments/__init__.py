from essai.arguments.data_args import *
from essai.arguments.model_args import *
from essai.arguments.training_args import *
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MetaArguments:
    name: Optional[str] = field(
        default=None, metadata={"help": "Model and Method name"}
    )
    description: Optional[str] = field(
        default=None, metadata={"help": "Description of the experiment"}
    )
    model_argument: Optional[str] = field(
        default=None, metadata={"help": "Model Huggingface arguments"}
    )
    dataset_argument: Optional[str] = field(
        default=None, metadata={"help": "dataset huggingface arguments"}
    )
    training_argument: Optional[str] = field(
        default=None, metadata={"help": "training arguments"}
    )


def build_arguments(config):
    if config.model_argument == "NI":
        model_args = NIModelArguments
    if config.dataset_argument == "NI":
        data_args = NIDataArguments
    if config.training_argument == "NI":
        training_args = NITrainingArguments
    return model_args, data_args, training_args, MetaArguments