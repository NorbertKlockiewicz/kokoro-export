import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from executorch.devtools import Inspector

etrecord_path = "debug/etrecord.bin"
etdump_path = "debug/etdump.etdp"
inspector = Inspector(etdump_path=etdump_path, etrecord=etrecord_path)

df = inspector.to_dataframe()

# Oblicz sumaryczny czas wszystkich metod
total_time = df.loc[2:1817, 'avg (ms)'].iloc[0]

# Oblicz sumaryczny czas dla native_call_convolution.out
conv_time = df[df['event_name'] == 'native_call_convolution.out']['avg (ms)'].sum()

# Oblicz udział procentowy
if total_time > 0:
    percent = 100 * conv_time / total_time
else:
    percent = 0

print(f"Share of 'native_call_convolution.out' in total execution time: {conv_time:.4f} ms / {total_time:.4f} ms = {percent:.2f}%")