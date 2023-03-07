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
    dtype=torch.bfloat16,
    injection_policy={T5Block: ('SelfAttention.o', 'EncDecAttention.o', 'DenseReluDense.wo')}
)
generator.device = torch.device(f'cuda:{local_rank}')

output = generator("""Definition: You need to read the given passage from a chapter of a textbook and and construct a 
                   multi-choices educational question based on a provided answer about information in the passage. 
                   The question needs to be answerable from the passage and also include possible answer choices. 
                   Input: Answer: socialized medicine. \nPassage: One critique of the Patient Protection and Affordable Care Act is that it will create a system of socialized medicine , a term that for many Americans has negative connotations lingering from the Cold War era and earlier . Under a socialized medicine system , the government owns and runs the system . It employs the doctors , nurses , and other staff , and it owns and runs the hospitals ( Klein 2009 ) . <hl> The best example of socialized medicine is in Great Britain , where the National Health System ( NHS ) gives free health care to all its residents . <hl> And despite some Americans \u2019 knee-jerk reaction to any health care changes that hint of socialism , the United States has one socialized system with the Veterans Health Administration.
                   Output: """
                )
if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    print(output)