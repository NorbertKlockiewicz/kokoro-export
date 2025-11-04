import torch
from executorch.runtime import Runtime

# Fixed seed for consistency
torch.manual_seed(410375)

runtime = Runtime.get()

# Sample input
text_sample = "Hello world!"
asr_example = torch.randn(1, 512, 62)
F0_pred_example = torch.randn(1, 124)
N_pred_example = torch.randn(1, 124)
ref_s_example = torch.randn(1, 128)

program = runtime.load_program("model.pte")
method = program.load_method("forward")
outputs = method.execute([asr_example, F0_pred_example, N_pred_example, ref_s_example])

for i, output in enumerate(outputs):
    if torch.isnan(output).any():
        print(f"Output {i} contains NaN values.")
    elif torch.isinf(output).any():
        print(f"Output {i} contains Inf values.")
    elif not torch.isfinite(output).all():
        print(f"Output {i} contains non-finite values.")
    else:
        print("Output - type:", output.dtype)
        print(output)