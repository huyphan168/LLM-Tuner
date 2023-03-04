""""
dataset builder
"""
from datasets import load_dataset
from essai.dataset.natural_instructions.ni_collator import DataCollatorForNI
from essai.dataset.flan.flan_collator import DataCollatorForFlan

def build_dataset(config, tokenizer, data_args, training_args, model_args, model):
    if config.dataset_argument == "NI":
        label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
        data_collator = DataCollatorForNI(
                        tokenizer,
                        model=model,
                        padding="max_length" if data_args.pad_to_max_length else "longest",
                        max_source_length=data_args.max_source_length,
                        max_target_length=data_args.max_target_length,
                        label_pad_token_id=label_pad_token_id,
                        pad_to_multiple_of=8 if training_args.fp16 else None,
                        add_task_name=data_args.add_task_name,
                        add_task_definition=data_args.add_task_definition,
                        num_pos_examples=data_args.num_pos_examples,
                        num_neg_examples=data_args.num_neg_examples,
                        add_explanation=data_args.add_explanation,
                        tk_instruct=data_args.tk_instruct
        )
        raw_datasets = load_dataset(
            data_args.data_file, 
            data_dir=data_args.data_dir, 
            task_dir=data_args.task_dir, 
            cache_dir=model_args.cache_dir,
            max_num_instances_per_task=data_args.max_num_instances_per_task,
            max_num_instances_per_eval_task=data_args.max_num_instances_per_eval_task
        )
    elif config.dataset_argument == "Flan":
        data_collator = DataCollatorForFlan(
                        tokenizer,
                        model=model,
                        padding="max_length" if data_args.pad_to_max_length else "longest",
                        max_source_length=data_args.max_source_length,
                        max_target_length=data_args.max_target_length,
                        pad_to_multiple_of=8 if training_args.fp16 else None,
                        add_few_shots=data_args.add_few_shots,
                        num_shots=data_args.num_shots
        )
        raw_datasets = load_dataset(
            data_args.data_file, 
            data_dir=data_args.data_dir, 
            task_dir=data_args.task_dir, 
            cache_dir=model_args.cache_dir
        )
    return data_collator, raw_datasets

