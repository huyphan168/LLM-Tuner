import os
import deepspeed
import torch
from transformers.models.t5.modeling_t5 import T5Block
from transformers import pipeline

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '4'))
generator = pipeline('text2text-generation', model="allenai/tk-instruct-3b-def-pos")



generator.model = deepspeed.init_inference(
    generator.model,
    mp_size = world_size,
    dtype=torch.float16,
    injection_policy={T5Block: ('SelfAttention.o', 'EncDecAttention.o', 'DenseReluDense.wo')}
)
generator.device = torch.device(f'cuda:{local_rank}')

output = generator("DeepSpeed is")
if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    print(output)