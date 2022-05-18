from pkgutil import iter_modules


def module_exists():
    """Check environments(tf or pytorch).

    Returns
    -------
    is_torch: bool
    """
    try:
        import torch.nn
        return True
    except:
        return False
