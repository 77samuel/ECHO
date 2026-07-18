"""
figures.py

Generates Figures 1-5 from the accompanying manuscript. Requires analysis.py
to have been run first (for the .npy files in --data-dir/../results, or pass
--results-dir directly), and the raw trace files to be available in --data-dir
for figures that recompute per-bin statistics.

Usage:
    python figures.py --data-dir ./data --results-dir ./results --out-dir ./figures
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from difflib import SequenceMatcher

from extraction import extract_evidence_terms, extract_probability, jaccard_overlap
from echo_metric import compute_echo, EPSILON

plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})


def _box(ax, x, y, w, h, text, color='#e8f0fe'):
    r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", facecolor=color, edgecolor='black', linewidth=1)
    ax.add_patch(r)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=8.5)


def _arrow(ax, x1, y1, x2, y2):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=15, color='black')
    ax.add_patch(a)


def figure1_echo_pipeline(out_path):
    """Figure 1: ECHO computation pipeline diagram."""
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.set_xlim(0, 10); ax.set_ylim(0, 2); ax.axis('off')
    stages = ['Trace\n(3 rounds)', 'Evidence\nextraction', 'Belief\nextraction', 'ECHO\ncomputation', 'Statistical\nanalysis']
    for i, s in enumerate(stages):
        _box(ax, i * 2, 0.5, 1.7, 1, s, '#fdf0d5')
        if i < 4:
            _arrow(ax, i * 2 + 1.7, 1, i * 2 + 2, 1)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def figure2_heatmap(records, out_path):
    """Figure 2: Evidence-change x belief-change heatmap."""
    jaccards = np.array([r['jaccard'] for r in records])
    prob_diffs = np.array([r['prob_diff'] for r in records])
    ev_changed = jaccards <= 0.35
    prob_changed = prob_diffs >= EPSILON

    data = np.array([
        [100 * np.mean((~ev_changed) & (~prob_changed)), 100 * np.mean((~ev_changed) & prob_changed)],
        [100 * np.mean(ev_changed & (~prob_changed)), 100 * np.mean(ev_changed & prob_changed)],
    ])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(data, cmap='YlOrRd', vmin=0, vmax=45)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Belief\nunchanged', 'Belief\nchanged'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['Same\nevidence', 'Different\nevidence'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{data[i, j]:.1f}%', ha='center', va='center', fontsize=13, fontweight='bold')
    ax.set_title(f'Evidence change x Belief change (n={len(records)})')
    plt.colorbar(im, label='% of traces')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def figure3_justification_evolution(peer_traces, out_path):
    """Figure 3: rationale text similarity across self-critique rounds."""
    sims_01, sims_12, sims_02 = [], [], []
    for t in peer_traces:
        calls = t['trace']['calls']
        r0, r1, r2 = calls[0]['response']['text'], calls[1]['response']['text'], calls[2]['response']['text']
        sims_01.append(SequenceMatcher(None, r0, r1).ratio())
        sims_12.append(SequenceMatcher(None, r1, r2).ratio())
        sims_02.append(SequenceMatcher(None, r0, r2).ratio())

    rounds = ['R0->R1', 'R1->R2 ref', 'R0->R2 (full)']
    vals = [np.mean(sims_01), np.mean(sims_12), np.mean(sims_02)]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(rounds, vals, color='#31a354')
    ax.set_ylabel('Text similarity (SequenceMatcher ratio)')
    ax.set_title('Rationale text similarity across self-critique rounds')
    ax.set_ylim(0, 0.3)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def figure4_probability_bins(records, out_path):
    """Figure 4: stated-belief stability by initial probability bin."""
    bins = [(i / 10, (i + 1) / 10) for i in range(10)]
    labels = [f'[{lo:.1f}-{hi:.1f})' for lo, hi in bins]
    pct_unchanged = []
    for lo, hi in bins:
        sub = [r for r in records if lo <= r['p0'] < hi]
        if sub:
            pct_unchanged.append(100 * np.mean([r['belief_unchanged'] for r in sub]))
        else:
            pct_unchanged.append(np.nan)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, pct_unchanged, color='#2c7fb8')
    ax.set_ylabel('% traces with unchanged probability')
    ax.set_xlabel('Initial probability bin')
    ax.set_title('Stated-belief stability by initial probability magnitude')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def figure5_cross_architecture(echo_vals, retention_vals, incorp_vals, out_path):
    """Figure 5: evidence-utilization measures across three architectures (real data)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    rng = np.random.default_rng(0)
    x1 = rng.normal(1, 0.06, len(echo_vals))
    x2 = rng.normal(2, 0.06, len(retention_vals))
    x3 = rng.normal(3, 0.06, len(incorp_vals))

    ax.scatter(x1, echo_vals, alpha=0.55, color='#e6550d', s=25, label=f'peer_critique_debate (ECHO), n={len(echo_vals)}')
    ax.scatter(x2, retention_vals, alpha=0.55, color='#3182bd', s=25, label=f'sequential_pipeline (retention), n={len(retention_vals)}')
    ax.scatter(x3, incorp_vals, alpha=0.55, color='#31a354', s=25, label=f'orchestrator_specialist (incorporation), n={len(incorp_vals)}')

    for x, data in [(1, echo_vals), (2, retention_vals), (3, incorp_vals)]:
        ax.hlines(data.mean(), x - 0.15, x + 0.15, color='black', linewidth=2, zorder=5)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['peer_critique_debate\n(ECHO)', 'sequential_pipeline\n(retention)', 'orchestrator_specialist\n(incorporation)'])
    ax.set_ylabel('Evidence-utilization measure (architecture-specific, NOT directly comparable)')
    ax.set_title('Evidence-utilization measures across three architectures')
    ax.legend(loc='upper right', fontsize=7)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='./data')
    parser.add_argument('--results-dir', default='./results')
    parser.add_argument('--out-dir', default='./figures')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from analysis import load_traces, build_peer_critique_records

    peer_traces = load_traces(os.path.join(args.data_dir, 'peer_critique_debate.jsonl'))
    records = build_peer_critique_records(peer_traces)

    figure1_echo_pipeline(os.path.join(args.out_dir, 'figure1_echo_pipeline.png'))
    figure2_heatmap(records, os.path.join(args.out_dir, 'figure2_heatmap.png'))
    figure3_justification_evolution(peer_traces, os.path.join(args.out_dir, 'figure3_justification_evolution.png'))
    figure4_probability_bins(records, os.path.join(args.out_dir, 'figure4_probability_bins.png'))

    echo_vals = np.load(os.path.join(args.results_dir, 'bjri_raw_CONFIRMED.npy'))
    retention_vals = np.load(os.path.join(args.results_dir, 'retention_raw_CONFIRMED.npy'))
    incorp_vals = np.load(os.path.join(args.results_dir, 'incorp_raw_FINAL.npy'))
    figure5_cross_architecture(echo_vals, retention_vals, incorp_vals, os.path.join(args.out_dir, 'figure5_cross_architecture.png'))

    print(f"All 5 figures written to {args.out_dir}/")


if __name__ == '__main__':
    main()
