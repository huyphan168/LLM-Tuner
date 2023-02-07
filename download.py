from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("togethercomputer/GPT-JT-6B-v1")
model = AutoModelForCausalLM.from_pretrained("togethercomputer/GPT-JT-6B-v1", from_tf=True)
