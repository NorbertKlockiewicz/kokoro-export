import spaces
from kokoro import KModel
from kokoro.pipeline import KPipeline
import gradio as gr
import os
import random
import torch
import matplotlib.pyplot as plt


CUDA_AVAILABLE = torch.cuda.is_available()
models = {gpu: KModel().to('cuda' if gpu else 'cpu').eval() for gpu in [False] + ([True] if CUDA_AVAILABLE else [])}
pipelines = {lang_code: KPipeline(lang_code=lang_code, model=False) for lang_code in 'ab'}
pipelines['a'].g2p.lexicon.golds['kokoro'] = 'kˈOkəɹO'
pipelines['b'].g2p.lexicon.golds['kokoro'] = 'kˈQkəɹQ'

@spaces.GPU(duration=30)
def forward_gpu(ps, ref_s, speed, target_tokens=None):
    return models[True](ps, ref_s, speed, target_tokens=target_tokens)

def generate_first(text, voice='af_heart', speed=1, use_gpu=CUDA_AVAILABLE, target_tokens=None):
    pipeline = pipelines[voice[0]]
    pack = pipeline.load_voice(voice)
    use_gpu = use_gpu and CUDA_AVAILABLE
    for _, ps, _ in pipeline(text, voice, speed):
        ref_s = pack[len(ps)-1]
        try:
            if use_gpu:
                return forward_gpu(ps, ref_s, speed, target_tokens=target_tokens)
            else:
                return models[False](ps, ref_s, speed, target_tokens=target_tokens)
        except gr.exceptions.Error as e:
            if use_gpu:
                gr.Warning(str(e))
                gr.Info('Retrying with CPU. To avoid this error, change Hardware to CPU.')
                return models[False](ps, ref_s, speed, target_tokens=target_tokens)
            else:
                raise gr.Error(e)
    return None


def estimate_durations_from_file(filepath, voice='af_heart', speed=1, use_gpu=CUDA_AVAILABLE, target_tokens=None):
    import statistics

    durations = []
    duration_counts = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    total = len(lines)
    print(f"Processing {total} lines...")
    for idx, line in enumerate(lines, 1):
        result = generate_first(line, voice=voice, speed=speed, use_gpu=use_gpu, target_tokens=target_tokens)
        try:
            duration = int(result)
            durations.append(duration)
            duration_counts[duration] = duration_counts.get(duration, 0) + 1
        except Exception as e:
            print(f"Error processing line {idx}: {e}")
        # Simple progress bar
        if idx % max(1, total // 50) == 0 or idx == total:
            bar_len = 40
            filled_len = int(bar_len * idx // total)
            bar = '=' * filled_len + '-' * (bar_len - filled_len)
            print(f"\r[{bar}] {idx}/{total}", end='', flush=True)
    print()  # Newline after progress bar

    if durations:
        mean_duration = statistics.mean(durations)
        min_duration = min(durations)
        max_duration = max(durations)
        print(f"\nStatistics for durations:")
        print(f"Mean: {mean_duration}")
        print(f"Min: {min_duration}")
        print(f"Max: {max_duration}")

        # Draw histogram and mark mean
        plt.figure(figsize=(8, 4))
        plt.hist(durations, bins=30, color='skyblue', edgecolor='black')
        plt.axvline(mean_duration, color='red', linestyle='dashed', linewidth=2, label=f"Mean: {mean_duration:.2f}")
        plt.title("Histogram of Durations")
        plt.xlabel("Duration")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("No durations measured.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Warning: No path to quotes file given.")
        print("Usage: python estimate_shapes.py <quotes_file.txt> [target_tokens]")
    elif len(sys.argv) < 3:
        print("Warning: No target_tokens argument given.")
        print("Usage: python estimate_shapes.py <quotes_file.txt> [target_tokens]")
        estimate_durations_from_file(sys.argv[1], target_tokens=None)
    else:
        target_tokens = int(sys.argv[2])
        estimate_durations_from_file(sys.argv[1], target_tokens=target_tokens)


