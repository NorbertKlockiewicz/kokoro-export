"""
Compare decoder output with randn_like vs randn implementation
to see if our replacement causes numerical instability
"""

import torch
from kokoro import KModel

print('=' * 80)
print('Comparing randn_like vs randn implementation')
print('=' * 80)

# Create test inputs with FIXED seed for reproducibility
torch.manual_seed(42)
asr = torch.randn(1, 512, 78)
F0_pred = torch.randn(1, 156)
N_pred = torch.randn(1, 156)
ref_s = torch.randn(1, 128)

print('\nTest inputs created with seed=42')
print(f'  asr: {asr.shape}, range [{asr.min():.3f}, {asr.max():.3f}]')
print(f'  F0_pred: {F0_pred.shape}, range [{F0_pred.min():.3f}, {F0_pred.max():.3f}]')
print(f'  N_pred: {N_pred.shape}, range [{N_pred.min():.3f}, {N_pred.max():.3f}]')
print(f'  ref_s: {ref_s.shape}, range [{ref_s.min():.3f}, {ref_s.max():.3f}]')

# Test 1: Current implementation (with our modifications)
print('\n' + '=' * 80)
print('1. Testing CURRENT implementation (deterministic sin/cos noise)')
print('=' * 80)

model_current = KModel(repo_id='hexgrad/Kokoro-82M', disable_complex=True).eval()

with torch.no_grad():
    torch.manual_seed(42)  # Reset seed
    audio_current = model_current.decoder(asr, F0_pred, N_pred, ref_s).squeeze()

    print(f'Output shape: {audio_current.shape}')
    print(f'Has NaN: {torch.isnan(audio_current).any()}')
    print(f'Has Inf: {torch.isinf(audio_current).any()}')
    print(f'Min: {audio_current.min():.6f}')
    print(f'Max: {audio_current.max():.6f}')
    print(f'Mean: {audio_current.mean():.6f}')
    print(f'Std: {audio_current.std():.6f}')

    # Check for extreme values
    extreme_threshold = 1e6
    extreme_count = (torch.abs(audio_current) > extreme_threshold).sum().item()
    if extreme_count > 0:
        print(f'⚠️  WARNING: {extreme_count} values exceed {extreme_threshold}!')
        print(f'   This suggests numerical instability')

# Now let's check what the noise values look like
print('\n' + '=' * 80)
print('2. Checking noise generation in SourceModuleHnNSF')
print('=' * 80)

# Get the sine waves to see what we're using as noise base
gen = model_current.decoder.generator
f0_upsamp = gen.f0_upsamp
m_source = gen.m_source

with torch.no_grad():
    torch.manual_seed(42)
    f0_up = f0_upsamp(F0_pred.unsqueeze(1)).transpose(1, 2)

    # Get sine_wavs before noise is added
    sine_wavs, uv, noise = m_source.l_sin_gen(f0_up)

    print(f'\nSine waves (before noise):')
    print(f'  Shape: {sine_wavs.shape}')
    print(f'  Range: [{sine_wavs.min():.6f}, {sine_wavs.max():.6f}]')
    print(f'  Mean: {sine_wavs.mean():.6f}')
    print(f'  Std: {sine_wavs.std():.6f}')

    print(f'\nNoise (deterministic):')
    print(f'  Shape: {noise.shape}')
    print(f'  Range: [{noise.min():.6f}, {noise.max():.6f}]')
    print(f'  Mean: {noise.mean():.6f}')
    print(f'  Std: {noise.std():.6f}')

    print(f'\nUV (voiced/unvoiced):')
    print(f'  Shape: {uv.shape}')
    print(f'  Range: [{uv.min():.6f}, {uv.max():.6f}]')
    print(f'  Mean: {uv.mean():.6f}')

# Test 3: Check the generator output (before ISTFT)
print('\n' + '=' * 80)
print('3. Checking Generator output (spec and phase before ISTFT)')
print('=' * 80)

with torch.no_grad():
    torch.manual_seed(42)

    # Run through decoder up to generator
    F0 = model_current.decoder.F0_conv(F0_pred.unsqueeze(1))
    N = model_current.decoder.N_conv(N_pred.unsqueeze(1))
    x = torch.cat([asr, F0, N], axis=1)
    x = model_current.decoder.encode(x, ref_s)

    # Decode
    asr_res = model_current.decoder.asr_res(asr)
    res = True
    for block in model_current.decoder.decode:
        if res:
            x = torch.cat([x, asr_res, F0, N], axis=1)
        x = block(x, ref_s)
        if block.upsample_type != "none":
            res = False

    # Before generator ISTFT - get spec and phase
    f0_gen = f0_upsamp(F0_pred.unsqueeze(1)).transpose(1, 2)
    har_source, noi_source, uv = gen.m_source(f0_gen)
    har_source = har_source.transpose(1, 2).squeeze(1)
    har_spec, har_phase = gen.stft.transform(har_source)
    har = torch.cat([har_spec, har_phase], dim=1)

    # Run through generator conv layers
    for i in range(gen.num_upsamples):
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.1)
        x_source = gen.noise_convs[i](har)
        x_source = gen.noise_res[i](x_source, ref_s)
        x = gen.ups[i](x)
        if i == gen.num_upsamples - 1:
            x = gen.reflection_pad(x)
        x = x + x_source
        xs = None
        for j in range(gen.num_kernels):
            if xs is None:
                xs = gen.resblocks[i*gen.num_kernels+j](x, ref_s)
            else:
                xs += gen.resblocks[i*gen.num_kernels+j](x, ref_s)
        x = xs / gen.num_kernels

    x = torch.nn.functional.leaky_relu(x)
    x = gen.conv_post(x)

    # Get spec and phase
    spec = torch.exp(x[:,:gen.post_n_fft // 2 + 1, :])
    phase = torch.sin(x[:, gen.post_n_fft // 2 + 1:, :])

    print(f'Spec (torch.exp output):')
    print(f'  Shape: {spec.shape}')
    print(f'  Range: [{spec.min():.6f}, {spec.max():.6f}]')
    print(f'  Mean: {spec.mean():.6f}')

    # Check for extreme exp values
    x_spec = x[:,:gen.post_n_fft // 2 + 1, :]
    print(f'\nPre-exp values (x for spec):')
    print(f'  Range: [{x_spec.min():.6f}, {x_spec.max():.6f}]')

    if x_spec.max() > 100:
        print(f'  ⚠️  WARNING: Pre-exp values exceed 100!')
        print(f'     torch.exp(100) = {torch.exp(torch.tensor(100.0)):.2e}')
        print(f'     This causes numerical overflow!')

    print(f'\nPhase (torch.sin output):')
    print(f'  Shape: {phase.shape}')
    print(f'  Range: [{phase.min():.6f}, {phase.max():.6f}]')

print('\n' + '=' * 80)
print('Analysis Complete')
print('=' * 80)
