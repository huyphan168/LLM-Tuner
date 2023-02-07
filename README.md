# Distributed Finetuning for Human-centered Writing Assistant
LehrerAI Language Model Team

## To do:
- [x] Refactor Tk-instruct code
- [x] Refactor `essai/train.py` to disengage arguments, Natural Instruction code Interaction for universality (include GPT-like or OPT models) steps: <Trainer, Dataset, Arguments(use Config instead)>
- [x] Refactor `essai/trainer` to include more training code
- [ ] Implement Flan-T5, GPT-like models fine-tuning code
- [ ] Collect data from books `essai/datasets/vspace_ielts/collect_books.py`
- [ ] Implement Triton fused kernel + torch.compile

## Installation
Current version of implementation has tested with pytorch 2.0 but it should work with pytorch > 1.0.0. To install all dependencies, just run.
```
python3 -r requirements.txt
```

## Usage
Essai is modular and flexible. It is easy to add new tasks, models, and algorithms. On top of different repo or codebase of different algorithms is Meta Essai Configs to choose which algorithm to use.

```yaml
## Meta Configs for Essai
name: t5-instruct-3b
description: T5 model with instructive prompts
model_argument: NI
dataset_argument: NI
training_argument: NI
## Huggingface Configs
run_name: t5-experiment
deepspeed: essai/ds_configs/stage3.config
do_train: True
do_predict: True
predict_with_generate: True
model_name_or_path: google/t5-xl-lm-adapt
data_file: essai/datasets/natural_instructions/ni_dataset.py
max_source_length: 1024
save_strategy: steps
save_steps: 2500
bf16: True
...
```
To reconstruct the results in Tk-instruct, just run
```
deepspeed --master_port $(shuf -i25000-30000 -n1) essai/train.py essau/configs/t5_instruct_3b.yaml
```
## Code Files
This repository provides the following Folders and Files
- `essai`: The main folder for the codebase
  - `arguments`: The folder for the arguments classes
  - `configs`: The folder for the configuration files
  - `datasets`: The folder for the dataset Folders
    - `natural_instructions`: The folder for the Natural Instruction dataset
    - `vspace_ielts`: The folder for the Vspace IELTS dataset
  - `trainers`: The folder for the trainer files for different algorithms
  - `inference`: The folder for the inference server using EnergonAI + Quantization Strategies
  - `ds_configs`: The folder for the deepspeed configuration files
  - `train.py`: The main file for training
- `notebooks`: The folder for the notebooks
- `scripts`: The folder for the bash scripts
- `requirements.txt`: The file for the dependencies
- `README.md`: The file for the README
## Citation
```bibtex
@misc{https://doi.org/10.48550/arxiv.2204.07705,
  doi = {10.48550/ARXIV.2204.07705},
  
  url = {https://arxiv.org/abs/2204.07705},
  
  author = {Wang, Yizhong and Mishra, Swaroop and Alipoormolabashi, Pegah and Kordi, Yeganeh and Mirzaei, Amirreza and Arunkumar, Anjana and Ashok, Arjun and Dhanasekaran, Arut Selvan and Naik, Atharva and Stap, David and Pathak, Eshaan and Karamanolakis, Giannis and Lai, Haizhi Gary and Purohit, Ishan and Mondal, Ishani and Anderson, Jacob and Kuznia, Kirby and Doshi, Krima and Patel, Maitreya and Pal, Kuntal Kumar and Moradshahi, Mehrad and Parmar, Mihir and Purohit, Mirali and Varshney, Neeraj and Kaza, Phani Rohitha and Verma, Pulkit and Puri, Ravsehaj Singh and Karia, Rushang and Sampat, Shailaja Keyur and Doshi, Savan and Mishra, Siddhartha and Reddy, Sujan and Patro, Sumanta and Dixit, Tanay and Shen, Xudong and Baral, Chitta and Choi, Yejin and Smith, Noah A. and Hajishirzi, Hannaneh and Khashabi, Daniel},
  
  keywords = {Computation and Language (cs.CL), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution 4.0 International}
}

```
