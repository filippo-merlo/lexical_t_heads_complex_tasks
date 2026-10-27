"""Pin a process to one GPU by its nvidia-smi index; call before torch is imported.

Without ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` CUDA numbers the GPUs fastest first and an index
can land on another user's card (head_level_jlens scripts/_gpu.py).
"""

import os


def pin(gpu: int) -> None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # Long batched generations grow the KV cache step by step; expandable segments keep
    # that from fragmenting GPU memory into out-of-memory retries.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
