import random
import numpy as np
import torch

def set_seed(seed=42):
    """
    Ensures full reproducibility across random, NumPy, PyTorch (CPU + CUDA), and cuDNN.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensures deterministic behavior for CuDNN (slower, but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"✅ Seed set: {seed}")