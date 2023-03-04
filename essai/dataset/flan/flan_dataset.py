import json
import os
import random
import datasets

logger = datasets.logging.get_logger(__name__)

_DESCRIPTION = """
Flan Collection, described in The Flan Collection: Designing Data and Methods for Effective Instruction Tuning 
and used to produce Flan-T5 and Flan-PaLM.
"""

class FlanConfig(datasets.BuilderConfig):
    """BuilderConfig for Flan Dataset Collection"""

    def __init__(self, data_dir=None, task_dir=None, max_num_instances_per_task=None, max_num_instances_per_eval_task=None, **kwargs):
        """BuilderConfig for EduQG.

        Args:
          data_dir: `string`, directory containing the data files
          task_dir: `string`, directory containing the task files
          **kwargs: keyword arguments forwarded to super.
        """
        super(FlanConfig, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.task_dir = task_dir

class FlanInstructions(datasets.GeneratorBasedBuilder):
    """Flan Dataset."""

    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIG_CLASS = FlanConfig
    BUILDER_CONFIGS = [
      FlanConfig(name="default", description="Default config for Flan")
    ]
    DEFAULT_CONFIG_NAME = "default"
    
    def _info(self):
      return datasets.DatasetInfo(
        description=_DESCRIPTION,
        features=datasets.Features(
          {
            "id": datasets.Value("string"),
            "Task": datasets.Value("string"),
            "Source": [datasets.Value("string")],
            "URL": [datasets.Value("string")],
            "Categories": [datasets.Value("string")],
            "Instruction Prompt": [datasets.Value("string")],
            "Examplars": [{
              "input": datasets.Value("string"),
              "output": datasets.Value("string")
            }],
            "Instance": {
              "id": datasets.Value("string"),
              "input": datasets.Value("string"),
              "output": [datasets.Value("string")]
            },
            "Instance License": [datasets.Value("string")]
          }
        ),
        supervised_keys=None,
        homepage="https://github.com/google-research/FLAN"
      )
      
    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        if self.config.data_dir is None or self.config.task_dir is None:
            self.config.data_dir = self.config.data_dir 
            self.config.task_dir = self.config.task_dir 

        split_dir = self.config.data_dir
        task_dir = self.config.task_dir

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "path": os.path.join(split_dir, "train_tasks.txt"), 
                    "task_dir": task_dir, 
                    "subset": "train"
                }),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "path": os.path.join(split_dir, "dev_tasks.txt"), 
                    "task_dir": task_dir,
                    "subset": "dev"
                }),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "path": os.path.join(split_dir, "test_tasks.txt"), 
                    "task_dir": task_dir, 
                    "subset": "test"
                }),
        ]

    def _generate_examples(self, path=None, task_dir=None, subset=None):
        """Yields examples."""
        logger.info(f"Generating tasks from = {path}")
        with open(path, encoding="utf-8") as split_f:
            for line in split_f:
                task_name = line.strip()
                task_path = os.path.join(task_dir, task_name + ".json")
                with open(task_path, encoding="utf-8") as task_f:
                    s = task_f.read()
                    task_data = json.loads(s)
                    task_data["Task"] = task_name
                    if "Instruction Source" in task_data:
                        task_data.pop("Instruction Source")
                    instances = task_data.pop("Instances")
                    for idx, instance in enumerate(instances):
                        example = task_data.copy()
                        example["id"] = instance["id"]
                        example["Instance"] = instance
                        yield f"{task_name}_{idx}", example