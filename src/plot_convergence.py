import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import glob
import os

# ── Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

CONDITION_STYLE = {
    'A_baseline': {
        'label': 'No PQC, No DP (Baseline)',
        'color': '#2196F3',   # blue
        'linestyle': '-',
        'marker': 'o',
        'markevery': 5,
    },
    'B_pqc_only': {
        'label': 'ML-KEM-512 (PQC only)',
        'color': '#4CAF50',   # green
        'linestyle': '--',
        'marker': 's',
        'markevery': 5,
    },
    'C_pqc_dp': {
        'label': 'ML-KEM-512 + DP-SGD (ε=EPSILON)',
        'color': '#F44336',   # red
        'linestyle': ':',
        'marker': '^',
        'markevery': 5,
    },
}

def extract_convergence(csv_path: str, epsilon: float = None) -> dict:
    """
    Reads benchmark CSV and returns per-round mean ± std global accuracy
    across all runs, for each condition. Returns dict keyed by condition name.
    """
    df = pd.read_csv(csv_path)

    # Keep only aggregate rows (per-client rows have client_id != 'aggregate')
    df = df[df['client_id'] == 'aggregate'].copy()
    df['server_round'] = df['server_round'].astype(int)
    df['global_accuracy'] = pd.to_numeric(df['global_accuracy'], errors='coerce')

    results = {}
    for condition in df['condition'].unique():
        cdf = df[df['condition'] == condition]
        grouped = cdf.groupby('server_round')['global_accuracy']
        mean = grouped.mean()
        std = grouped.std().fillna(0)
        results[condition] = {'mean': mean, 'std': std}

    return results

def plot_convergence(
    csv_path: str,
    output_path: str,
    title: str,
    dataset_label: str,
    epsilon: float,
    architecture_label: str,
):
    data = extract_convergence(csv_path)

    fig, ax = plt.subplots(figsize=(8, 5))

    for condition, style in CONDITION_STYLE.items():
        if condition not in data:
            continue

        mean = data[condition]['mean']
        std = data[condition]['std']
        rounds = mean.index

        # Replace EPSILON placeholder in C label
        label = style['label'].replace('EPSILON', str(epsilon))

        ax.plot(
            rounds, mean,
            label=label,
            color=style['color'],
            linestyle=style['linestyle'],
            linewidth=2,
            marker=style['marker'],
            markevery=style['markevery'],
            markersize=5,
        )
        ax.fill_between(
            rounds,
            mean - std,
            mean + std,
            alpha=0.15,
            color=style['color'],
        )

    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Global Accuracy')
    ax.set_title(title)
    ax.set_xlim(1, rounds.max())
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.legend(loc='lower right', framealpha=0.9)

    # Annotation box with key stats
    textstr = f'Dataset: {dataset_label}\nArchitecture: {architecture_label}\nClients: 2  |  Rounds: 40  |  Runs: 5'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.4)
    ax.text(0.02, 0.97, textstr, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    base = os.path.expanduser("~/pq-flids/results")

    # ── Figure 1: ToN-IoT ─────────────────────────────────────────────
    toniot_csv = os.path.join(base, "benchmark_20260311_175951.csv")
    plot_convergence(
        csv_path=toniot_csv,
        output_path=os.path.join(base, "convergence_curves_toniot.png"),
        title="Convergence — ToN-IoT Dataset (1D-CNN + FedAvg)",
        dataset_label="ToN-IoT (4 classes, 19 features)",
        epsilon=1.263,
        architecture_label="1D-CNN",
    )

    # ── Figure 2: CICIoT2023 ──────────────────────────────────────────
    # The three 5-run × 40-round CSVs for conditions A, B, C (LayerNorm runs)
    ciciot_files = [
        os.path.join(base, "benchmark_ciciot2023_20260315_175141.csv"),  # A_baseline
        os.path.join(base, "benchmark_ciciot2023_20260316_095053.csv"),  # B_pqc_only
        os.path.join(base, "benchmark_ciciot2023_20260316_223043.csv"),  # C_pqc_dp
    ]

    if all(os.path.exists(f) for f in ciciot_files):
        combined = pd.concat([pd.read_csv(f) for f in ciciot_files], ignore_index=True)
        combined_path = os.path.join(base, "benchmark_ciciot2023_combined.csv")
        combined.to_csv(combined_path, index=False)
        print(f"Combined {len(ciciot_files)} CSVs → {combined_path}")

        plot_convergence(
            csv_path=combined_path,
            output_path=os.path.join(base, "convergence_curves_ciciot2023.png"),
            title="Convergence — CICIoT2023 Dataset (MobileNetV2-1D + FedProx)",
            dataset_label="CICIoT2023 (7 classes, 39 features)",
            epsilon=0.912,
            architecture_label="MobileNetV2-1D (LayerNorm)",
        )
    else:
        missing = [f for f in ciciot_files if not os.path.exists(f)]
        print(f"ERROR: Missing CICIoT2023 CSVs: {missing}")
