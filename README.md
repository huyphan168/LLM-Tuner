# Distributed Finetuning for Education
LehrerAI Language Model Team
---
## To do:
- [x] Refactor Tk-instruct code
- [x] Refactor `essai/train.py` to disengage arguments, Natural Instruction code Interaction for universality (include GPT-like or OPT models) steps: <Trainer, Dataset, Arguments(use Config instead)>
- [x] Refactor `essai/trainer` to include more training code
- [ ] Implement Flan-T5, GPT-like models fine-tuning code
- [ ] Collect data from books 
- [ ] Implement Triton fused kernel + torch.compile
---
## Installation
Current version of implementation has tested with pytorch 2.0 nightly but it should work with pytorch > 1.0.0. To install dependencies, just run.

```
conda env create --file=environment.yaml
```
Notes that the installation of deepspeed might encounter multiple errors, the most common one is `fused_adam` optimizer in deepspeed needs to be compiled by CUDA toolkit that has the same version with binary
cuda that attached with pytorch. For example, the cuda version of pytorch is 11.7 then the CUDA toolkit to be installed will have version of 11.7. We can manually install CUDA toolkit by instruction of NVIDIA. 
However, we can install the toolkit via cuda easily by
```
conda install cuda -c nvidia/label/cuda-11.7.0
```
Other errors related to compilation can be addressed by aliasing the .so file of cuda in conda environment folder to the shared folder such as `/usr/local/lib`. You can check whether the compilation is successful or not:
```
python3 make_cpuadam.py
```
If the compilation need file cuda profiler since your toolkit does not have it. You can copy paste `cuda_profiler_api.h` to `/usr/local/your_cuda/include`.
Other erros like `cannot import inf from torch._six`, please see https://github.com/microsoft/DeepSpeed/pull/2863.

---
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
To reconstruct the results in Tk-instruct of EduQG on both tasks `question generation` and `options generation, we first to format the original dataset EduQG to follow instruction + input -> output format of Tk-instruct
```bash
python3 data/eduqg/prepare_eduqg_instruct.py
```
This will generate data in folder of `data/eduqg/tasks/`, for this case we have two tasks. The folder `data/eduqg/splits/defaults` will split which tasks and files will be evaluated on. After generating the data, we can fine-tune `Tk-instruct 3B` on
our dataset by running:
```bash
deepspeed train.py essai/configs/t5_instruct_3b_qog_pos.yaml
```
To test ability of the model, please edit the prompt according to the data format mentioned in `data/eduqg/prepare_eduqg_instruct.py` and run this file with your checkpoint
```bash
python3 test_deepspeed.py
```

---
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
- `notebooks`: The folder for the notebooks
- `scripts`: The folder for the bash scripts
- `requirements.txt`: The file for the dependencies
- `README.md`: The file for the README
- `train.py`: The main file for training

## Extend the Library to other instructions
 - `essai/arguments`:  By adding your algorithm's arguments
    - `data_args.py`: add data arguments
    - `model_args.py`: add model arguments
    - `training_args.py`: add training arguments for trainer.
 - `essai/configs/t5_flan_xl_qog_pos.yaml`: add configuration for training
 - `essai/dataset/flan`: add new dataset format for tuning
    - `flan_collator.py`
    - `flan_dataset.py`
 - `essai/trainer/flan_trainer.py`: add trainer representing your algorithm

---

## Result on EduQG two tasks
I will denote `QG` for question generation task given passage and answer, `Opt` for options generation given question, passage and answers. And QOG for generating full multi-choices question.
I found that by seperating the QOG into 2 tasks and train them together boost the performance further. This might be due to the complexity of unified tasks (future can be address by Hindsight Relabling Instruction Tuning)
| Method | Tasks | rougeL | rouge1 |
| --- | --- | --- | --- |
| Tk-instruct Large | QOG | 33.3 | 37.34 |
| Tk-instruct 3B  | QOG | 32.49 | 36.7 |
| Tk-instruct 3B Def+Pos | QG | 43.48 | 48.36 | 
| Tk-instruct 3B Def+Pos Muti-tasks | QG | 43.79 | 48.58 |
| Tk-instruct 3B Def+Pos Muti-tasks | Opt | 45.92 | 51.24 |

----
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
