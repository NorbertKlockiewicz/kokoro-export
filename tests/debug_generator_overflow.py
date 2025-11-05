"""
Debug Generator to find where overflow occurs
"""

import torch
from kokoro import KModel

print("Debugging Generator overflow")
print("=" * 80)

# Load model
model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True).eval()

# Load captured inputs
data = torch.load('generator_test_inputs.pt')
x = data['x']
s = data['s']
f0 = data['f0']

print(f"Inputs:")
print(f"  x: {x.shape}, range: [{x.min():.3f}, {x.max():.3f}]")
print(f"  s: {s.shape}, range: [{s.min():.3f}, {s.max():.3f}]")
print(f"  f0: {f0.shape}, range: [{f0.min():.3f}, {f0.max():.3f}]")

# Manually run through Generator to find overflow
gen = model.decoder.generator

with torch.no_grad():
    # SourceModule processing
    f0_up = gen.f0_upsamp(f0[:, None]).transpose(1, 2)
    print(f"\nAfter f0_upsamp: {f0_up.shape}, range: [{f0_up.min():.3f}, {f0_up.max():.3f}]")

    har_source, noi_source, uv = gen.m_source(f0_up)
    print(f"har_source: {har_source.shape}, range: [{har_source.min():.3f}, {har_source.max():.3f}]")

    har_source_t = har_source.transpose(1, 2).squeeze(1)
    har_spec, har_phase = gen.stft.transform(har_source_t)
    print(f"har_spec: {har_spec.shape}, range: [{har_spec.min():.3f}, {har_spec.max():.3f}]")
    print(f"har_phase: {har_phase.shape}, range: [{har_phase.min():.3f}, {har_phase.max():.3f}]")

    har = torch.cat([har_spec, har_phase], dim=1)
    print(f"har: {har.shape}, range: [{har.min():.3f}, {har.max():.3f}]")

    # Upsampling blocks
    x_current = x
    for i in range(gen.num_upsamples):
        x_current = torch.nn.functional.leaky_relu(x_current, negative_slope=0.1)
        x_source = gen.noise_convs[i](har)
        x_source = gen.noise_res[i](x_source, s)
        x_current = gen.ups[i](x_current)
        if i == gen.num_upsamples - 1:
            x_current = gen.reflection_pad(x_current)
        x_current = x_current + x_source

        xs = None
        for j in range(gen.num_kernels):
            if xs is None:
                xs = gen.resblocks[i * gen.num_kernels + j](x_current, s)
            else:
                xs += gen.resblocks[i * gen.num_kernels + j](x_current, s)
        x_current = xs / gen.num_kernels

        print(f"\nAfter upsample block {i}: {x_current.shape}, range: [{x_current.min():.3f}, {x_current.max():.3f}]")

    # Final conv
    x_current = torch.nn.functional.leaky_relu(x_current)
    x_current = gen.conv_post(x_current)
    print(f"\nAfter conv_post: {x_current.shape}, range: [{x_current.min():.3f}, {x_current.max():.3f}]")

    # This is where overflow happens
    spec = torch.exp(x_current[:, : gen.post_n_fft // 2 + 1, :])
    print(f"\nAfter exp: {spec.shape}, range: [{spec.min():.3f}, {spec.max():.3f}]")
    print(f"  Contains Inf: {spec.isinf().any()}")

    phase = torch.sin(x_current[:, gen.post_n_fft // 2 + 1 :, :])
    print(f"phase: {phase.shape}, range: [{phase.min():.3f}, {phase.max():.3f}]")

    # ISTFT
    audio = gen.stft.inverse(spec, phase)
    print(f"\nFinal audio: {audio.shape}, range: [{audio.min():.3f}, {audio.max():.3f}]")

print("\n" + "=" * 80)
