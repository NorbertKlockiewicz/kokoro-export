import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from executorch.devtools import Inspector
import ast
import tabulate
from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)

ARTIFACTS_PATH = "profiling/artifacts"
etrecord_path = f"{ARTIFACTS_PATH}/etrecord.bin"
etdump_path = f"{ARTIFACTS_PATH}/etdump.etdp"
inspector = Inspector(etdump_path=etdump_path, etrecord=etrecord_path)

df = inspector.to_dataframe()

def _to_list(v):
    # Normalize 'raw' values to a list[float]
    if isinstance(v, (list, tuple)):
        return list(v)
    if isinstance(v, str):
        try:
            parsed = ast.literal_eval(v)
            if isinstance(parsed, (list, tuple)):
                return [float(x) for x in parsed]
            return [float(parsed)]
        except Exception:
            return []
    if pd.isna(v):
        return []
    try:
        return [float(v)]
    except Exception:
        return []

# 1) Per-op total time and share of Method::execute
exe_df = df[df['event_block_name'] == 'Execute'].copy()
exe_df['raw_list'] = exe_df['raw'].apply(_to_list)
exe_df['op_time_ms'] = exe_df['raw_list'].apply(lambda xs: sum(float(x) for x in xs))
exe_df['call_count'] = exe_df['raw_list'].apply(len)
exe_df['delegated_calls'] = exe_df.apply(
    lambda r: r['call_count'] if bool(r.get('is_delegated_op', False)) else 0, axis=1
)

stats = exe_df.groupby('event_name', as_index=False).agg(
    total_time_ms=('op_time_ms', 'sum'),
    total_calls=('call_count', 'sum'),
    delegated_calls=('delegated_calls', 'sum'),
)

method_mask = df['event_name'].astype(str).str.fullmatch(r'Method:{1,2}execute')
method_rows = df[method_mask]
total_execute_time_ms = (
    sum(sum(_to_list(v)) for v in method_rows['raw'])
    if not method_rows.empty
    else float(stats['total_time_ms'].sum())
)

stats['share_pct'] = (stats['total_time_ms'] / total_execute_time_ms * 100.0) if total_execute_time_ms > 0 else 0.0

# 2) Delegation counts/ratio per operation
stats['delegated_ratio_pct'] = (stats['delegated_calls'] / stats['total_calls'] * 100.0).fillna(0.0)

stats = stats.sort_values('total_time_ms', ascending=False)

def color_row(row):
    # Method::execute (and variants), DELEGATE_CALL, OPERATOR_CALL should remain uncolored
    event_name = str(row['event_name'])
    is_method_execute = event_name.startswith("Method:") or event_name.startswith("Method::")
    is_special = event_name in ("DELEGATE_CALL", "OPERATOR_CALL")
    base = ""
    if is_method_execute or is_special:
        base = Style.BRIGHT  # white/bold, no color
    elif row['delegated_ratio_pct'] == 0:
        base = Fore.RED + Style.BRIGHT
    elif row['delegated_ratio_pct'] == 100:
        base = Fore.GREEN + Style.BRIGHT
    elif 0 < row['delegated_ratio_pct'] < 100:
        base = Fore.YELLOW + Style.BRIGHT
    else:
        base = ""
    # Top 3 by time: keep bold, but don't override Method::execute or special coloring
    if not (is_method_execute or is_special):
        if row['name'] == 0:
            base += Style.BRIGHT
        elif row['name'] == 1:
            base += Style.BRIGHT
        elif row['name'] == 2:
            base += Style.BRIGHT
    # Remove 'name' from output columns
    values = [
        row['event_name'],
        f"{row['total_time_ms']:.3f}",
        f"{row['share_pct']:.2f}%",
        int(row['total_calls']),
        int(row['delegated_calls']),
        f"{row['delegated_ratio_pct']:.2f}%",
    ]
    return [base + str(x) + Style.RESET_ALL for x in values]

table_data = []
for idx, row in stats.iterrows():
    formatted = [
        row['event_name'],
        f"{row['total_time_ms']:.3f}",
        f"{row['share_pct']:.2f}%",
        int(row['total_calls']),
        int(row['delegated_calls']),
        f"{row['delegated_ratio_pct']:.2f}%",
    ]
    table_data.append(formatted)

# Add header styling
headers = [
    Fore.BLUE + Style.BRIGHT + "Op Name" + Style.RESET_ALL,
    Fore.BLUE + Style.BRIGHT + "Total Time (ms)" + Style.RESET_ALL,
    Fore.BLUE + Style.BRIGHT + "Share (%)" + Style.RESET_ALL,
    Fore.BLUE + Style.BRIGHT + "Calls" + Style.RESET_ALL,
    Fore.BLUE + Style.BRIGHT + "Delegated" + Style.RESET_ALL,
    Fore.BLUE + Style.BRIGHT + "Delegated (%)" + Style.RESET_ALL,
]

# Color rows
styled_table = []
for i, row in enumerate(stats.itertuples(index=False)):
    row_dict = row._asdict()
    row_dict['name'] = i
    styled_table.append(color_row(row_dict))

print(Fore.YELLOW + Style.BRIGHT + "\nPer-op Execute stats (ms and delegation):\n" + Style.RESET_ALL)
print(tabulate.tabulate(styled_table, headers=headers, tablefmt="fancy_grid", stralign="right", numalign="right"))