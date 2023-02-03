#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/home/ubuntu/20huy.pn/.cache/huggingface

port=$(shuf -i25000-30000 -n1)

deepspeed --master_port $port essai/train.py essai/configs/t5_instruct_3b.yaml
