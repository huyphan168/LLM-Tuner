# Distributed Finetuning for Human-centered Writing Assistant
LehrerAI Language Model Team

## To do:
- [x] Refactor Tk-instruct code
- [ ] Refactor `essai/train.py` to disengage arguments, Natural Instruction code Interaction for universality (include GPT-like or OPT models) steps: <Trainer, Dataset, Arguments(use Config instead)>
- [ ] Refactor `essai/trainer` to include more training code
- [ ] Implement Flan-T5, GPT-like models fine-tuning code
- [ ] Equip repo with CollosalAI code instead of only DeepSpeed for better converage of distributed training algorithms
- [ ] Collect data from books `essai/datasets/vspace_ielts/collect_books.py`

## Installation
Current version of implementation has tested with pytorch 2.0 but it should work with pytorch > 1.0.0. To install all dependencies, just run.
```
python3 -r requirements.txt
```

## Usage
This implementation is fairly simple to use. With a user specificed configuration file, `main.py` will do anything else. The configuration file consists of 2 parts: 
`task` to handle problem-specific parameters and `model` to handle model-specific parameters. For example, the configuration file for DeepThinking on Mazes dataset is as follows.

```yaml
seed: 0
task: 
    task_type: mazes # (for general task, in future may be more useful)
    dataset: mazes
    batch_size: 40
    num_workers: 4
    num_epochs: 100
    lr: 0.001
    weight_decay: 0.0001
    log_interval: 100
    save_interval: 1000
    save_dir: ./checkpoints/dt_mazes
    device: cuda
model:
    architecture: deepthinking
    backbone:
        backbone_type: deepthinking2d
        in_channels: 3
        width: 128
    recurrence:
        recur_type: deepthinking2d
        num_blocks: [2]
        group_norm: False
        in_channels: 128
        width: 128
        max_iter_fwd: 30
    decoder:
        decoder_type: deepthinking2d
        width: 128
```
To reconstruct the results in the paper of DeepThinking without Overthinking with Mazes dataset, just run
```
python3 main.py --cfg configs/deepthinking_mazes.yaml --output-path results/dt2d_mazes
```
## Code Files
This repository provides the following Folders and Files
- `configs/` : Configuration files for different tasks.
- `results/` : Folder to save results.
- `scripts/` : Folder to save scripts for running on cluster or experiments.
- `main.py` : Main file to run the code.
- `evaluation.py`: Evaluation file to evaluate the model depends on user specificed tasks.
- `src/backbones` : Implementation of different backbones including DVAE, ResNet, projectConV for feature extraction.
- `src/data` : Implementation of CLEVR, Mazes, Chess, Graphs
- `src/utils.py` : Implementation of some useful functions.
- `src/decoder.py`: Implementation of decoder for needed tasks.
- `src/tasks.py`: TaskExecutor class to run modules specificed by certain problems including computing metrics, losses or preprocess data.
- `src/model.py` : Implementation of Main models including DEQ, DeepThinking, RIM, DynamicMoEC


## Citation
```bibtex

```
