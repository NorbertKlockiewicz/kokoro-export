import torch


def indices_to_counters(indices: torch.Tensor) -> torch.Tensor:
    num_classes = int(indices.max().item()) + 1
    counters = torch.zeros(num_classes, dtype=torch.long)
    for idx in indices:
        counters[idx] += 1

    return counters


def counters_to_indices(b: torch.Tensor) -> torch.Tensor:
    indices = []
    for idx, count in enumerate(b):
        indices.extend([idx] * int(count.item()))
        
    return torch.tensor(indices, dtype=torch.long)


# A simple heuristic to "scale" integer-based indice duration tensor to match the target duration
def scale(indices: torch.Tensor, target_len: int,
          up: bool = True) -> torch.Tensor:
    a = indices_to_counters(indices)

    # NOTE: a is a 1D tensor of shape [N]
    factor = target_len / a.sum().item()

    x = a * factor
    b = torch.floor(x).long() if up else torch.ceil(x).long()

    remaining = abs(target_len - b.sum().item())

    diff: torch.Tensor = torch.abs(x - b)
    diff_table = [(idx, value) for idx, value in enumerate(diff)]
    diff_table.sort(key=lambda x: x[1], reverse=True)

    for i in range(remaining):
      b[i] += 1 if up else -1
    
    return counters_to_indices(b)
