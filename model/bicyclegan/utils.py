def ckpt_convert(param):
    return {
        'module.' + k: v
        for k, v in param.items()
    }