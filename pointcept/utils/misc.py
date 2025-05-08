import os
import warnings
from collections import abc
import numpy as np
import torch
from importlib import import_module


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def intersection_and_union(output, target, K, ignore_index=-1):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape

    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)

    mask = target != ignore_index  # Exclude ignore_index elements
    output = output[mask]
    target = target[mask]

    tp_mask = output == target  # True Positive mask
    fp_mask = output != target  # False Positive or False Negative mask

    area_intersection, _ = np.histogram(output[tp_mask], bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection

    # Calculate FP and FN
    area_fp = area_output - area_intersection
    area_fn = area_target - area_intersection

    # TN: Pixels that are neither in the prediction nor in the ground truth
    total_pixels = mask.sum()
    area_tn = total_pixels - (area_fp + area_fn + area_intersection)

    return area_intersection, area_union, area_target, area_fp, area_fn, area_tn

def intersection_and_union_gpu(output, target, k, ignore_index=-1):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape

    output = output.view(-1)
    target = target.view(-1)

    mask = target != ignore_index  # Exclude ignore_index elements
    output = output[mask]
    target = target[mask]

    tp_mask = output == target  # True Positive mask
    fp_mask = output != target  # False Positive or False Negative mask

    area_intersection = torch.histc(output[tp_mask], bins=k, min=0, max=k - 1)
    area_output = torch.histc(output, bins=k, min=0, max=k - 1)
    area_target = torch.histc(target, bins=k, min=0, max=k - 1)
    area_union = area_output + area_target - area_intersection

    # Calculate FP and FN
    area_fp = area_output - area_intersection
    area_fn = area_target - area_intersection

    # TN: Pixels that are neither in the prediction nor in the ground truth
    total_pixels = mask.sum().item()
    area_tn = total_pixels - (area_fp + area_fn + area_intersection)

    return area_intersection, area_union, area_target, area_fp, area_fn, area_tn


def make_dirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)


def find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def is_seq_of(seq, expected_type, seq_type=None):

    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def is_str(x):

    return isinstance(x, str)


def import_modules_from_strings(imports, allow_failed_imports=False):

    if not imports:
        return
    single_import = False
    if isinstance(imports, str):
        single_import = True
        imports = [imports]
    if not isinstance(imports, list):
        raise TypeError(f"custom_imports must be a list but got type {type(imports)}")
    imported = []
    for imp in imports:
        if not isinstance(imp, str):
            raise TypeError(f"{imp} is of type {type(imp)} and cannot be imported.")
        try:
            imported_tmp = import_module(imp)
        except ImportError:
            if allow_failed_imports:
                warnings.warn(f"{imp} failed to import and is ignored.", UserWarning)
                imported_tmp = None
            else:
                raise ImportError
        imported.append(imported_tmp)
    if single_import:
        imported = imported[0]
    return imported


class DummyClass:
    def __init__(self):
        pass
