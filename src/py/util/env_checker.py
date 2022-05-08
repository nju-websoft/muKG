from pkgutil import iter_modules


def module_exists():
    try:
        import torch.nn
        return True
    except:
        return False
