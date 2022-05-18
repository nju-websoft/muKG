import time
import numpy as np

from src.py.util.env_checker import module_exists


def to_var(batch, device):
    """Change an array to a variable, and put it to the specific device(CPU OR GPU).
    """
    from torch.autograd import Variable
    import torch
    return Variable(torch.from_numpy(np.array(batch)).to(device))


def to_tensor(batch, device):
    """Change an array to a tensor, and put it to the specific device(CPU OR GPU).
    """
    import torch
    a = np.array(batch)
    a = torch.from_numpy(a)
    return a.to(device)


def to_tensor_cpu(batch):
    import torch
    return torch.from_numpy(np.array(batch))


def load_session():
    import tensorflow._api.v2.compat.v1 as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = '0,1,2'
    return tf.Session(config=config)


def merge_dic(dic1, dic2):
    return {**dic1, **dic2}


def early_stop(flag1, flag2, flag):
    """Terminate model training by checking if accuracy drops.
    """
    if flag <= flag2 <= flag1:
        print("\n == should early stop == \n")
        return flag2, flag, True
    else:
        return flag2, flag, False


def task_divide(idx, n):
    """Divide ids into n steps.

    Returns
    -------
    tasks: list
        Specify id list for each task.
    """
    total = len(idx)
    if n <= 0 or 0 == total:
        return [idx]
    if n > total:
        return [idx]
    elif n == total:
        return [[i] for i in idx]
    else:
        j = total // n
        tasks = []
        for i in range(0, (n - 1) * j, j):
            tasks.append(idx[i:i + j])
        tasks.append(idx[(n - 1) * j:])
        return tasks


def generate_out_folder(out_folder, training_data_path, div_path, method_name):
    params = training_data_path.strip('/').split('/')
    print(out_folder, training_data_path, params, div_path, method_name)
    path = params[-1]
    if module_exists():
        envs = "torch"
    else:
        envs = "tf"
    folder = out_folder + method_name + '/' + path + "/" + div_path + "/" + envs + "/"
    print("results output folder:", folder)
    return folder


def parse_resources(res):
    try:
        device, number = res.split(':')
        return device, int(number)
    except:
        raise Exception("Invalid input of resources")