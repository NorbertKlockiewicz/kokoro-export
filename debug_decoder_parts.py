"""
Debug decoder parts to find where NaN is generated
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
print('Debugging Decoder Parts - Finding NaN Source')
print('=' * 80)

# Load model
model = KModel(repo_id='hexgrad/Kokoro-82M', disable_complex=True).eval()

# Create test inputs
asr = torch.randn(1, 512, 78)
F0_pred = torch.randn(1, 156)
N_pred = torch.randn(1, 156)
ref_s = torch.randn(1, 128)

print('\nTesting decoder parts individually...\n')

# ============================================================================
# Part 1: Test SourceModuleHnNSF (F0 to harmonic source)
# ============================================================================
print('1. Testing SourceModuleHnNSF (F0 processing)...')
print('-' * 80)

class SourceModuleTest(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.m_source = model.decoder.generator.m_source
        self.f0_upsamp = model.decoder.generator.f0_upsamp

    def forward(self, F0_curve):
        f0 = self.f0_upsamp(F0_curve.unsqueeze(1)).transpose(1, 2)
        har_source, noi_source, uv = self.m_source(f0)
        return har_source, noi_source, uv

source_test = SourceModuleTest(model)

with torch.no_grad():
    # Test in PyTorch
    har_pt, noi_pt, uv_pt = source_test(F0_pred)
    print(f'  PyTorch output: har {har_pt.shape}, noi {noi_pt.shape}, uv {uv_pt.shape}')
    print(f'  Has NaN: har={torch.isnan(har_pt).any()}, noi={torch.isnan(noi_pt).any()}, uv={torch.isnan(uv_pt).any()}')

    # Export to ExecuTorch
    exported = export(source_test, (F0_pred,), strict=False)

    # Check for randn
    randn_count = sum(1 for node in exported.graph.nodes if node.op == 'call_function' and 'randn' in str(node.target).lower())
    print(f'  randn ops: {randn_count}')

    try:
        edge = to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()])
        exec_prog = edge.to_executorch()

        os.makedirs('exported_pte/debug', exist_ok=True)
        pte_path = 'exported_pte/debug/source_module.pte'
        with open(pte_path, 'wb') as f:
            f.write(exec_prog.buffer)

        # Test ExecuTorch
        runtime = Runtime.get()
        program = runtime.load_program(Path(pte_path), verification=Verification.Minimal)
        method = program.load_method('forward')

        output_et = method.execute((F0_pred,))
        har_et, noi_et, uv_et = output_et

        print(f'  ExecuTorch output: har {har_et.shape}, noi {noi_et.shape}, uv {uv_et.shape}')
        print(f'  Has NaN: har={torch.isnan(har_et).any()}, noi={torch.isnan(noi_et).any()}, uv={torch.isnan(uv_et).any()}')

        if torch.isnan(har_et).any() or torch.isnan(noi_et).any() or torch.isnan(uv_et).any():
            print('  ❌ FOUND NaN in SourceModuleHnNSF!')
        else:
            print('  ✓ SourceModuleHnNSF OK')

    except Exception as e:
        print(f'  ✗ Export failed: {str(e)[:100]}')

print()

# ============================================================================
# Part 2: Test CustomSTFT transform (forward)
# ============================================================================
print('2. Testing CustomSTFT.transform (forward STFT)...')
print('-' * 80)

class STFTForwardTest(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.stft = model.decoder.generator.stft

    def forward(self, audio):
        magnitude, phase = self.stft.transform(audio)
        return magnitude, phase

stft_test = STFTForwardTest(model)
test_audio = torch.randn(1, 10000)

with torch.no_grad():
    # Test in PyTorch
    mag_pt, phase_pt = stft_test(test_audio)
    print(f'  PyTorch output: mag {mag_pt.shape}, phase {phase_pt.shape}')
    print(f'  Has NaN: mag={torch.isnan(mag_pt).any()}, phase={torch.isnan(phase_pt).any()}')

    # Export
    exported = export(stft_test, (test_audio,), strict=False)

    try:
        edge = to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()])
        exec_prog = edge.to_executorch()

        pte_path = 'exported_pte/debug/stft_forward.pte'
        with open(pte_path, 'wb') as f:
            f.write(exec_prog.buffer)

        # Test ExecuTorch
        runtime = Runtime.get()
        program = runtime.load_program(Path(pte_path), verification=Verification.Minimal)
        method = program.load_method('forward')

        output_et = method.execute((test_audio,))
        mag_et, phase_et = output_et

        print(f'  ExecuTorch output: mag {mag_et.shape}, phase {phase_et.shape}')
        print(f'  Has NaN: mag={torch.isnan(mag_et).any()}, phase={torch.isnan(phase_et).any()}')

        if torch.isnan(mag_et).any() or torch.isnan(phase_et).any():
            print('  ❌ FOUND NaN in STFT forward!')
        else:
            print('  ✓ STFT forward OK')

    except Exception as e:
        print(f'  ✗ Export failed: {str(e)[:100]}')

print()

# ============================================================================
# Part 3: Test CustomSTFT inverse
# ============================================================================
print('3. Testing CustomSTFT.inverse (inverse STFT)...')
print('-' * 80)

class STFTInverseTest(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.stft = model.decoder.generator.stft

    def forward(self, magnitude, phase):
        return self.stft.inverse(magnitude, phase)

istft_test = STFTInverseTest(model)
test_mag = torch.randn(1, 11, 100).abs()  # magnitude should be positive
test_phase = torch.randn(1, 11, 100)

with torch.no_grad():
    # Test in PyTorch
    audio_pt = istft_test(test_mag, test_phase)
    print(f'  PyTorch output: {audio_pt.shape}')
    print(f'  Has NaN: {torch.isnan(audio_pt).any()}')

    # Export
    exported = export(istft_test, (test_mag, test_phase), strict=False)

    try:
        edge = to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()])
        exec_prog = edge.to_executorch()

        pte_path = 'exported_pte/debug/stft_inverse.pte'
        with open(pte_path, 'wb') as f:
            f.write(exec_prog.buffer)

        # Test ExecuTorch
        runtime = Runtime.get()
        program = runtime.load_program(Path(pte_path), verification=Verification.Minimal)
        method = program.load_method('forward')

        output_et = method.execute((test_mag, test_phase))
        audio_et = output_et[0]

        print(f'  ExecuTorch output: {audio_et.shape}')
        print(f'  Has NaN: {torch.isnan(audio_et).any()}')

        if torch.isnan(audio_et).any():
            print('  ❌ FOUND NaN in STFT inverse!')
            # Check specific values
            if torch.isnan(audio_et).all():
                print('     ALL values are NaN')
            else:
                nan_pct = torch.isnan(audio_et).sum().item() / audio_et.numel() * 100
                print(f'     {nan_pct:.1f}% of values are NaN')
        else:
            print('  ✓ STFT inverse OK')

    except Exception as e:
        print(f'  ✗ Export failed: {str(e)[:100]}')

print()

# ============================================================================
# Part 4: Test Generator (without STFT inverse, just the neural network part)
# ============================================================================
print('4. Testing Generator conv layers (without final STFT)...')
print('-' * 80)

class GeneratorNoSTFTTest(nn.Module):
    def __init__(self, model):
        super().__init__()
        gen = model.decoder.generator
        self.m_source = gen.m_source
        self.f0_upsamp = gen.f0_upsamp
        self.stft = gen.stft
        self.ups = gen.ups
        self.resblocks = gen.resblocks
        self.noise_convs = gen.noise_convs
        self.noise_res = gen.noise_res
        self.conv_post = gen.conv_post
        self.reflection_pad = gen.reflection_pad
        self.num_kernels = gen.num_kernels
        self.num_upsamples = gen.num_upsamples
        self.post_n_fft = gen.post_n_fft

    def forward(self, x, s, f0):
        with torch.no_grad():
            f0_up = self.f0_upsamp(f0.unsqueeze(1)).transpose(1, 2)
            har_source, noi_source, uv = self.m_source(f0_up)
            har_source = har_source.transpose(1, 2).squeeze(1)
            har_spec, har_phase = self.stft.transform(har_source)
            har = torch.cat([har_spec, har_phase], dim=1)

        for i in range(self.num_upsamples):
            x = torch.nn.functional.leaky_relu(x, negative_slope=0.1)
            x_source = self.noise_convs[i](har)
            x_source = self.noise_res[i](x_source, s)
            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)
            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x, s)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x, s)
            x = xs / self.num_kernels
        x = torch.nn.functional.leaky_relu(x)
        x = self.conv_post(x)
        spec = torch.exp(x[:,:self.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1:, :])
        # Return spec and phase WITHOUT calling inverse STFT
        return spec, phase

gen_test = GeneratorNoSTFTTest(model)

# Use proper input from decoder encode
with torch.no_grad():
    # Get the encoded input from decoder
    F0 = model.decoder.F0_conv(F0_pred.unsqueeze(1))
    N = model.decoder.N_conv(N_pred.unsqueeze(1))
    x_enc = torch.cat([asr, F0, N], axis=1)
    test_x = model.decoder.encode(x_enc, ref_s)

    # Test in PyTorch
    spec_pt, phase_pt = gen_test(test_x, ref_s, F0_pred)
    print(f'  PyTorch output: spec {spec_pt.shape}, phase {phase_pt.shape}')
    print(f'  Has NaN: spec={torch.isnan(spec_pt).any()}, phase={torch.isnan(phase_pt).any()}')

    # Export
    exported = export(gen_test, (test_x, ref_s, F0_pred), strict=False)

    try:
        edge = to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()])
        exec_prog = edge.to_executorch()

        pte_path = 'exported_pte/debug/generator_no_istft.pte'
        with open(pte_path, 'wb') as f:
            f.write(exec_prog.buffer)

        # Test ExecuTorch
        runtime = Runtime.get()
        program = runtime.load_program(Path(pte_path), verification=Verification.Minimal)
        method = program.load_method('forward')

        output_et = method.execute((test_x, ref_s, F0_pred))
        spec_et, phase_et = output_et

        print(f'  ExecuTorch output: spec {spec_et.shape}, phase {phase_et.shape}')
        print(f'  Has NaN: spec={torch.isnan(spec_et).any()}, phase={torch.isnan(phase_et).any()}')

        if torch.isnan(spec_et).any() or torch.isnan(phase_et).any():
            print('  ❌ FOUND NaN in Generator (before ISTFT)!')
        else:
            print('  ✓ Generator (without ISTFT) OK')

    except Exception as e:
        print(f'  ✗ Export failed: {str(e)[:100]}')

print()
print('=' * 80)
print('Debug complete. Check results above to identify NaN source.')
print('=' * 80)
