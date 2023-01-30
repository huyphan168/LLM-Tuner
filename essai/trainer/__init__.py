from essai.trainer.ni_trainer import NITrainer, DenserEvalCallback
from essai.datasets.compute_metrics import compute_metrics, compute_grouped_metrics
import numpy as np
import os
import json

def build_trainer(config, model, training_args, train_dataset, eval_dataset, tokenizer, data_collator):
    if config.meta.training_argument == "NI":

        def compute_ni_metrics(dataset, preds, save_prefix=None):
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            references = [e["Instance"]["output"] for e in dataset]
            result = compute_metrics(predictions=decoded_preds, references=references)
            result_per_task = compute_grouped_metrics(predictions=decoded_preds, references=references, groups=dataset["Task"])
            result.update(result_per_task)
            categories = ["_".join(it[0].lower().split()) for it in dataset["Categories"]]
            result_per_category = compute_grouped_metrics(predictions=decoded_preds, references=references, groups=categories)
            result.update(result_per_category)
            prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
            result["gen_len"] = np.mean(prediction_lens)
            result = {k: round(v, 4) for k, v in result.items()}
            if save_prefix is not None:
                with open(os.path.join(training_args.output_dir, f"{save_prefix}_eval_predictions.jsonl"), "w") as fout:
                    for example, pred in zip(dataset, decoded_preds):
                        fout.write(json.dumps({
                            "Task": example["Task"],
                            "Definition": example["Definition"],
                            "Instance": example["Instance"],
                            "Prediction": pred
                        }) + "\n")
            return result

        trainer = NITrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_ni_metrics if training_args.predict_with_generate else None,
        callbacks=[DenserEvalCallback] if training_args.denser_evaluation else None
    )
    return trainer