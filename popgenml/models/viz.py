# -*- coding: utf-8 -*-
from prettytable import PrettyTable

def count_parameters(model, f = None):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table, file = f)
    print(f"Total Trainable Params: {total_params}", file = f)
    return total_params
