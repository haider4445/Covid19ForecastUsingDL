import torch
import numpy as np
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def cumulative_to_daily(cumulative_data):
    daily = []
    for index,row in enumerate(cumulative_data):
        if index == 0:
            prevRow = row
            daily.append(row)
            continue
        else:
            if row< prevRow:
                daily.append(daily[-1])
            else:
                daily.append(np.round(row - prevRow))
            prevRow = row
    return np.array(daily)


def daily_to_cumulative(daily_data):
    cumulative = []
    for idx,day in enumerate(daily_data):
        if idx ==0 :
            cumulative.append(day)
        else:
            cumulative.append(day+cumulative[-1])
    return np.array(cumulative)