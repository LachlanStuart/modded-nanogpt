"""Interactive-only code for analysing the difference in model parameter statistics."""

assert str(0) != "0", "Don't run/import this file. Copy-paste into IPython & inspect df"

import torch
import re
import pandas as pd
from lovely_tensors import monkey_patch

monkey_patch() # This uses lovely_tensors' stringification for easily getting tensor summary stats.

state_dicts = {
    "NoSSMax": dict(torch.load("logs/20250204_NoSSMax320M/state_step020000.pt")['model']),
    "SSMax": dict(torch.load("logs/20250204_SSMax320M/state_step020000.pt")['model']),
    "SSMaxVoid": dict(torch.load("logs/20250211_SSMaxVoid320M/state_step020000.pt")['model']),
}

df = pd.DataFrame()
for model_name, state_dict in state_dicts.items():
    for key, tensor in state_dict.items():
        key = key.replace('_orig_mod.', '').replace('blocks.', '').replace("attn.lambdas", "attn_lambda").replace(".weight", "_w")
        key = ".".join(
            f"{int(part):02}" if part.isnumeric() else part for part in key.split('.')[::-1]
        )

        df.loc[key, [f"r_{model_name}",f"m_{model_name}",f"s_{model_name}"]] = (
            re.findall(r"(x∈\[.*\])? (μ=[^ ]+) (σ=[^ ]+)", str(tensor))[0]
        )

df = df.sort_index().sort_index(axis=1)

