import torch

def count_changed_weights( pre, post ):

    pre = [p for p in pre]
    post = [p for p in post]
    trues = 0
    trues_size = 0
    falses =0
    false_size = 0
    for p1, p2 in zip(pre, post):
        if torch.equal(p1, p2):
            trues_size += p1.numel()
            trues += 1
        else:
            false_size += p1.numel()
            falses += 1

    print ( f"Trues {trues} size {trues_size} Falses {falses} size {false_size}" )
