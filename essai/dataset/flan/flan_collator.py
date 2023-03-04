import logging
import random
import string
from transformers.data.data_collator import *

logger = logging.getLogger(__name__)


@dataclass
class DataCollatorForFlan:

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_source_length: Optional[int] = None
    max_target_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    add_few_shots: bool = False
    num_shots: int = 0
    text_only: bool=False
    

    def __call__(self, batch, return_tensors=None):

        if return_tensors is None:
            return_tensors = self.return_tensors

        sources = []
        for instance in batch:
            num_shots = self.num_shots
            add_few_shots = self.add_few_shots

            task_input = ""
            # add the input first.
            task_input += "Now complete the following example \n"
            task_input += f"{instance['Instance']['input'].strip()}"
            if not task_input[-1] in string.punctuation:
                task_input += "."
            task_input += "\n"
            task_input += "Output: "
            
            instruction_prompt = ""
            if isinstance(instance["Instruction Prompt"], list):
                instruction_prompt = instance["Instruction Prompt"][0].strip()
            else:
                instruction_prompt = instance["Instruction Prompt"].strip()
            
            # try to add positive examples.
            examplars = []
            if add_few_shots:
                for idx, example in enumerate(instance["Examplars"][:num_shots]):
                    example_str = f"({idx+1}) "
                    example_str += f"{example['input'].strip()}"
                    if not example_str[-1] in string.punctuation:
                        example_str += "."
                    example_str += "\n"
                    example_str += f" Output: {example['output'].strip()}"
                    if not example_str[-1] in string.punctuation:
                        example_str += "."
                    example_str += "\n" 
                    if len(self.tokenizer(instruction_prompt.format(examplars="".join(examplars), task_input=task_input) + example_str)["input_ids"]) <= self.max_source_length:
                        examplars.append(example_str)
                    else:
                        break
        
            source = instruction_prompt.format(examplars="".join(examplars), task_input=task_input)
            # print(source)
            tokenized_source = self.tokenizer(source)["input_ids"]
            if len(tokenized_source) <= self.max_source_length:
                sources.append(source)
            else:
                sources.append(self.tokenizer.decode(tokenized_source[:self.max_source_length], skip_special_tokens=True))

        if self.text_only:
            model_inputs = {"inputs": sources}
        else:
            model_inputs = self.tokenizer(
                sources, 
                max_length=self.max_source_length, 
                padding=self.padding,
                return_tensors=self.return_tensors, 
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of)

        if "output" in batch[0]["Instance"] and batch[0]["Instance"]["output"]:
            # Randomly select one reference if multiple are provided.
            labels = [random.choice(ex["Instance"]["output"]) for ex in batch]
            if self.text_only:
                model_inputs["labels"] = labels
            else:
                with self.tokenizer.as_target_tokenizer():
                    labels = self.tokenizer(
                        labels,
                        max_length=self.max_target_length,
                        padding=self.padding,
                        return_tensors=self.return_tensors,
                        truncation=True,
                        pad_to_multiple_of=self.pad_to_multiple_of
                    )
                label_mask = labels["attention_mask"].bool()
                model_inputs["labels"] = labels["input_ids"].masked_fill(~label_mask, self.label_pad_token_id)
        else:
            model_inputs["labels"] = None

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels") and not self.text_only:
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=model_inputs["labels"])
            model_inputs["decoder_input_ids"] = decoder_input_ids
            
        return model_inputs