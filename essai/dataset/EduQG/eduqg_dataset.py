import json
import os
import random
import datasets

logger = datasets.logging.get_logger(__name__)

class EduQGConfig(datasets.BuilderConfig):
    """BuilderConfig for EduQG."""

    def __init__(self, data_dir=None, task_dir=None, max_num_instances_per_task=None, max_num_instances_per_eval_task=None, **kwargs):
        """BuilderConfig for EduQG.

        Args:
          data_dir: `string`, directory containing the data files
          task_dir: `string`, directory containing the task files
          max_num_instances_per_task: `int`, maximum number of instances per task
          max_num_instances_per_eval_task: `int`, maximum number of instances per task for evaluation
          **kwargs: keyword arguments forwarded to super.
        """
        super(EduQGConfig, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.task_dir = task_dir
        self.max_num_instances_per_task = max_num_instances_per_task
        self.max_num_instances_per_eval_task = max_num_instances_per_eval_task

class EduQGInstructions(datasets.GeneratorBasedBuilder):
    """EduQGInstructions Dataset."""

    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIG_CLASS = EduQGConfig