""""
dataset builder
"""

from essai.dataset.natural_instructions.ni_collator import DataCollatorForNI

def build_collator(config, tokenizer, data_args, training_args, model):
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
    return data_collator
