"""
Test full decoder to find NaN source
"""

import torch
from torch import nn
from torch.export import export
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.runtime import Runtime, Verification
from pathlib import Path
import os

from kokoro import KModel

print('=' * 80)
print('Testing Full Decoder')
print('=' * 80)

# Load model
model = KModel(repo_id='hexgrad/Kokoro-82M', disable_complex=True).eval()

# Create test inputs
asr = torch.randn(1, 512, 78)
F0_pred = torch.randn(1, 156)
N_pred = torch.randn(1, 156)
ref_s = torch.randn(1, 128)

class DecoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.decoder = model.decoder

    def forward(self, asr, F0_pred, N_pred, ref_s):
        return self.decoder(asr, F0_pred, N_pred, ref_s).squeeze()

decoder = DecoderWrapper(model)

print('\n1. Testing decoder in PyTorch...')
print('-' * 80)

with torch.no_grad():
    # Test in PyTorch
    audio_pt = decoder(asr, F0_pred, N_pred, ref_s)
    print(f'  PyTorch output: {audio_pt.shape}')
    print(f'  Has NaN: {torch.isnan(audio_pt).any()}')
    if not torch.isnan(audio_pt).any():
        print(f'  Min: {audio_pt.min():.6f}, Max: {audio_pt.max():.6f}')

print('\n2. Exporting to ExecuTorch...')
print('-' * 80)

with torch.no_grad():
    # Export
    exported = export(decoder, (asr, F0_pred, N_pred, ref_s), strict=False)
    print('  ✓ PyTorch export successful')

    # Check for problematic ops
    randn_ops = []
    unfold_ops = []
    for node in exported.graph.nodes:
        if node.op == 'call_function':
            target = str(node.target)
            if 'randn' in target.lower():
                randn_ops.append(target)
            if 'unfold' in target.lower():
                unfold_ops.append(target)

    if randn_ops:
        print(f'  ⚠ Found {len(randn_ops)} randn ops: {set(randn_ops)}')
    else:
        print(f'  ✓ No randn ops')

    if unfold_ops:
        print(f'  ⚠ Found {len(unfold_ops)} unfold ops: {set(unfold_ops)}')
    else:
        print(f'  ✓ No unfold ops')

    # Convert to edge
    edge = to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()])
    print('  ✓ Edge conversion successful')

    # Convert to executorch
    exec_prog = edge.to_executorch()
    print('  ✓ ExecuTorch conversion successful')

    # Save
    os.makedirs('exported_pte', exist_ok=True)
    pte_path = 'exported_pte/decoder_debug.pte'
    with open(pte_path, 'wb') as f:
        f.write(exec_prog.buffer)

    size_mb = os.path.getsize(pte_path) / (1024 * 1024)
    print(f'  ✓ Saved to {pte_path} ({size_mb:.2f} MB)')

print('\n3. Testing decoder in ExecuTorch...')
print('-' * 80)

# Load and test ExecuTorch
runtime = Runtime.get()
program = runtime.load_program(Path(pte_path), verification=Verification.Minimal)
method = program.load_method('forward')

output_et = method.execute((asr, F0_pred, N_pred, ref_s))
audio_et = output_et[0]

print(f'  ExecuTorch output: {audio_et.shape}')
print(f'  Has NaN: {torch.isnan(audio_et).any()}')

if torch.isnan(audio_et).any():
    nan_count = torch.isnan(audio_et).sum().item()
    nan_pct = nan_count / audio_et.numel() * 100
    print(f'  ❌ NaN count: {nan_count} / {audio_et.numel()} ({nan_pct:.1f}%)')

    if torch.isnan(audio_et).all():
        print(f'     ALL values are NaN!')

    print('\n  Debugging NaN source:')
    print('  - SourceModuleHnNSF: OK (tested individually)')
    print('  - CustomSTFT forward: OK (tested individually)')
    print('  - CustomSTFT inverse: OK (tested individually)')
    print('  - Problem must be in: Generator conv/resblocks or XNNPACK backend')

else:
    print(f'  ✓ No NaN! Min: {audio_et.min():.6f}, Max: {audio_et.max():.6f}')

    # Compare with PyTorch
    mse = torch.mean((audio_pt - audio_et) ** 2).item()
    mae = torch.mean(torch.abs(audio_pt - audio_et)).item()
    print(f'\n  Comparison with PyTorch:')
    print(f'    MSE: {mse:.6f}')
    print(f'    MAE: {mae:.6f}')

print('\n' + '=' * 80)
print('Debug complete')
print('=' * 80)
