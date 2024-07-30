import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.3f}M'.format(tot / 1e6)
        elif tot >= 1e3:
            return '{:.2f}K'.format(tot / 1e3)
        else:
            return '{}'.format(tot)
    else:
        return tot
    

def get_names_dict(model):
    """Recursive walk to get names including path."""
    names = {}

    def _get_names(module, parent_name=""):
        for key, m in module.named_children():
            cls_name = str(m.__class__).split(".")[-1].split("'")[0]
            num_named_children = len(list(m.named_children()))
            if num_named_children > 0:
                name = parent_name + "." + key if parent_name else key
            else:
                name = parent_name + "." + cls_name + "_"+ key if parent_name else key
            names[name] = m

            if isinstance(m, nn.Module):
                _get_names(m, parent_name=name)

    _get_names(model)
    return names

# https://github.com/chenbong/ARM-Net/blob/main/utils/util.py
def get_model_flops(model, x, *args, **kwargs):
    """Summarize the given input model.
    Summarized information are 1) output shape, 2) kernel shape,
    3) number of the parameters and 4) operations (Mult-Adds)
    Args:
        model (Module): Model to summarize
        x (Tensor): Input tensor of the model with [N, C, H, W] shape
                    dtype and device have to match to the model
        args, kwargs: Other argument used in `model.forward` function
    """
    model.eval()
    if hasattr(model, 'module'):
        model = model.module
    #x = torch.zeros(input_size).to(next(model.parameters()).device)

    def register_hook(module):
        def hook(module, inputs, outputs):
            cls_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            key = None
            for name, item in module_names.items():
                if item == module:
                    key = "{}_{}".format(module_idx, name)
                    break
            assert key

            info = OrderedDict()
            info["id"] = id(module)
            if isinstance(outputs, (list, tuple)):
                try:
                    info["out"] = list(outputs[0].size())
                except AttributeError:
                    info["out"] = list(outputs[0].data.size())
            else:
                info["out"] = list(outputs.size())

            info["ksize"] = "-"
            info["inner"] = OrderedDict()
            info["params_nt"], info["params"], info["flops"] = 0, 0, 0

            for name, param in module.named_parameters():
                info["params"] += param.nelement() * param.requires_grad
                info["params_nt"] += param.nelement() * (not param.requires_grad)

                if name == "weight":
                    ksize = list(param.size())
                    if len(ksize) > 1:
                        ksize[0], ksize[1] = ksize[1], ksize[0]
                    info["ksize"] = ksize

                    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                        assert len(inputs[0].size()) == 4 and len(inputs[0].size()) == len(outputs[0].size())+1

                        in_c, in_h, in_w = inputs[0].size()[1:]
                        k_h, k_w = module.kernel_size
                        out_c, out_h, out_w = outputs[0].size()
                        groups = module.groups
                        kernel_mul = k_h * k_w * (in_c // groups)

                        kernel_mul_group = kernel_mul * out_h * out_w * (out_c // groups)
                        total_mul = kernel_mul_group * groups
                        info["flops"] += 2 * total_mul * inputs[0].size()[0] # total

                    elif isinstance(module, nn.BatchNorm2d):
                        info["flops"] += 2 * inputs[0].numel()

                    elif isinstance(module, nn.InstanceNorm2d):
                        info["flops"] += 6 * inputs[0].numel()

                    elif isinstance(module, nn.LayerNorm):
                        info["flops"] += 8 * inputs[0].numel()

                    elif isinstance(module, nn.Linear):
                        q = inputs[0].numel() // inputs[0].shape[-1]
                        info["flops"] += 2*q * module.in_features * module.out_features # total

                    elif isinstance(module, nn.PReLU) or isinstance(module, nn.ReLU):
                        info["flops"] += inputs[0].numel()
                    else:
                        print('not supported:', module)
                        exit()
                        info["flops"] += param.nelement()

                elif "weight" in name:
                    info["inner"][name] = list(param.size())
                    info["flops"] += param.nelement()

            if list(module.named_parameters()):
                for v in summary.values():
                    if info["id"] == v["id"]:
                        info["params"] = "(recursive)"

            #if info["params"] == 0:
            #    info["params"], info["flops"] = "-", "-"
            summary[key] = info

        if not module._modules:
            hooks.append(module.register_forward_hook(hook))

    module_names = get_names_dict(model)
    hooks = []
    summary = OrderedDict()

    model.apply(register_hook)
    try:
        with torch.no_grad():
            model(x) if not (kwargs or args) else model(x, *args, **kwargs)
    finally:
        for hook in hooks:
            hook.remove()
    # Use pandas to align the columns
    df = pd.DataFrame(summary).T

    df["Mult-Adds"] = pd.to_numeric(df["flops"], errors="coerce")
    df["Params"] = pd.to_numeric(df["params"], errors="coerce")
    df["Non-trainable params"] = pd.to_numeric(df["params_nt"], errors="coerce")
    df = df.rename(columns=dict(
        ksize="Kernel Shape",
        out="Output Shape",
    ))
    return df['Mult-Adds'].sum()
    '''
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        df_sum = df.sum()

    df.index.name = "Layer"

    df = df[["Kernel Shape", "Output Shape", "Params", "Mult-Adds"]]
    max_repr_width = max([len(row) for row in df.to_string().split("\n")])
    return df_sum["Mult-Adds"]
    '''