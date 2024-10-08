import sys
sys.path.append('yourpath/projects/Trace')
sys.path.append('yourpath/projects/Trace/trace')

# from Trace.trace.mistral_npu_monkey_patch import (
#     replace_with_torch_npu_flash_attention,
#     replace_with_torch_npu_rmsnorm
# )

# replace_with_torch_npu_flash_attention()
# replace_with_torch_npu_rmsnorm()

from Trace.trace.train_mt import train
import torch_npu
from torch_npu.contrib import transfer_to_npu

if __name__ == "__main__":
    train()