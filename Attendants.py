import torch
import torchvision.transforms as T
from PIL import Image
    
def select_device(info=False):
    """Select GPU or CPU device for torch
    """
    if torch.cuda.is_available():
        num_gpu = torch.cuda.device_count()
        if info:
            print(f"Detected {num_gpu} GPU.")

        for i in range(num_gpu):
            gpu_name = torch.cuda.get_device_name(i)
            if info:
                print(f"GPU {i}: {gpu_name}")

        # select device (GPU ar CPU)
        device = torch.device("cuda:0")
        if info:
            print(f"Selected device: {device}")
    else:
        device = torch.device("cpu")
        if info:
            print(f"GPU is unavailable. The {device} is being used.")
    
    return device

def get_resize():
    return T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.BICUBIC),
                    T.ToTensor()])