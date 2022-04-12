import torch


def get_device(cpu_only=False):
    has_gpu = torch.cuda.is_available()
    if cpu_only or not has_gpu:
        return torch.device('cpu')
    else:
        return torch.device('cuda')
