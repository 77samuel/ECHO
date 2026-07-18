"""
analysis.py

Reproduces all statistical results reported in Section 4 of the accompanying
manuscript. Requires the three ForesightFlow trace files (see README.md for
the data source) placed in a local `data/` directory:

    data/peer_critique_debate.jsonl
    data/sequential_pipeline.jsonl
    data/orchestrator_specialist.jsonl

Usage:
    python analysis.py --data-dir ./data --out-dir ./results
"""

import argparse
import json
import os

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from difflib import SequenceMatcher

from extraction import extract_evidence_terms, extract_probability, jaccard_overlap, STOPLIST_ORCHESTRATOR
from echo_metric import compute_echo, compute_evidence_retention, compute_evidence_incorporation, EPSILON


def load_traces(path):
    with open(path) as f:
        traces = [json.loads(line) for line in f]
    return [t for t in traces if len(t['trace']['calls']) == 3]


def build_peer_critique_records(traces):
    """Build per-trace records for the peer_critique_debate configuration."""
    records = []
    for t in traces:
        calls = t['trace']['calls']
        r0, r2 = calls[0]['response']['text'], calls[2]['response']['text']
        result = compute_echo(r0, r2, t['probability'])
        if result is None:
            continue
        sim = SequenceMatcher(None, r0, r2).ratio()
        result.update({
            'marketIndex': t['marketIndex'],
            'question': t.get('question', ''),
            'outcome': t.get('outcome'),
            'sim': sim,
        })
        records.append(result)
    return records


def table3_ablation(records, out):
    """Non-circular ablation: does evidence dissimilarity predict belief_unchanged alone?"""
    belief_unchanged = np.array([r['belief_unchanged'] for r in records])
    jaccard_dissim = np.array([1 - r['jaccard'] for r in records])
    text_dissim = np.array([1 - r['sim'] for r in records])

    r1, p1 = stats.pointbiserialr(belief_unchanged, jaccard_dissim)
    r2, p2 = stats.pointbiserialr(belief_unchanged, text_dissim)

    out['table3_ablation'] = {
        'filtered_evidence_dissimilarity': {'r': r1, 'p': p1},
        'unfiltered_text_dissimilarity': {'r': r2, 'p': p2},
    }


def table4_8_crosstab_and_association(records, out):
    """Evidence-change x belief-change crosstab, Fisher's exact test, Spearman, phi."""
    jaccard_threshold = 0.35
    jaccards = np.array([r['jaccard'] for r in records])
    prob_diffs = np.array([r['prob_diff'] for r in records])

    ev_changed = (jaccards <= jaccard_threshold).astype(int)
    prob_changed = (prob_diffs >= EPSILON).astype(int)

    a = int(np.sum((ev_changed == 1) & (prob_changed == 1)))
    b = int(np.sum((ev_changed == 1) & (prob_changed == 0)))
    c = int(np.sum((ev_changed == 0) & (prob_changed == 1)))
    d = int(np.sum((ev_changed == 0) & (prob_changed == 0)))

    odds_ratio, p_fisher = stats.fisher_exact([[a, b], [c, d]])
    n = a + b + c + d
    chi2 = n * (a * d - b * c) ** 2 / ((a + b) * (c + d) * (a + c) * (b + d))
    phi = np.sqrt(chi2 / n)

    spearman_rho, spearman_p = stats.spearmanr(1 - jaccards, prob_diffs)

    out['table4_crosstab'] = {'a': a, 'b': b, 'c': c, 'd': d}
    out['table8_association'] = {
        'odds_ratio': odds_ratio, 'fisher_p': p_fisher, 'phi': phi,
        'spearman_rho': spearman_rho, 'spearman_p': spearman_p,
    }


def table9_10_sensitivity(records, out):
    """Jaccard threshold sensitivity (Table 9) and epsilon sensitivity (Table 10)."""
    jaccards = np.array([r['jaccard'] for r in records])
    prob_diffs = np.array([r['prob_diff'] for r in records])

    jaccard_results = {}
    for thresh in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
        ev_changed = (jaccards <= thresh).astype(int)
        prob_changed = (prob_diffs >= EPSILON).astype(int)
        a = int(np.sum((ev_changed == 1) & (prob_changed == 1)))
        b = int(np.sum((ev_changed == 1) & (prob_changed == 0)))
        c = int(np.sum((ev_changed == 0) & (prob_changed == 1)))
        d = int(np.sum((ev_changed == 0) & (prob_changed == 0)))
        if min(a, b, c, d) > 0:
            odds_ratio, p = stats.fisher_exact([[a, b], [c, d]])
            jaccard_results[thresh] = {'odds_ratio': odds_ratio, 'p': p}

    epsilon_results = {}
    for eps in [0, 0.005, 0.01, 0.02, 0.05]:
        pct_unchanged = 100 * np.mean(prob_diffs <= eps)
        epsilon_results[eps] = pct_unchanged

    out['table9_jaccard_sensitivity'] = jaccard_results
    out['table10_epsilon_sensitivity'] = epsilon_results


def table11_permutation_test(records, out, n_perm=5000, seed=123):
    """Permutation test for the evidence-dissimilarity / probability-change association."""
    rng = np.random.default_rng(seed)
    jaccards = np.array([r['jaccard'] for r in records])
    prob_diffs = np.array([r['prob_diff'] for r in records])
    dissim = 1 - jaccards

    observed_rho = stats.spearmanr(dissim, prob_diffs)[0]
    perm_rhos = np.array([
        stats.spearmanr(dissim, rng.permutation(prob_diffs))[0] for _ in range(n_perm)
    ])
    p_perm = np.mean(np.abs(perm_rhos) >= np.abs(observed_rho))

    out['table11_permutation'] = {
        'observed_rho': observed_rho,
        'null_mean': perm_rhos.mean(),
        'null_sd': perm_rhos.std(),
        'p_value': p_perm,
    }


def table12_outcome_conditioned(records, out, n_boot=5000, seed=7):
    """ECHO by eventual outcome, with rank-biserial effect size and bootstrap CI."""
    rng = np.random.default_rng(seed)
    echo = np.array([r['echo'] for r in records])
    outcome = np.array([r['outcome'] for r in records])

    echo_correct = echo[outcome == 1]
    echo_wrong = echo[outcome == 0]

    u_stat, p_val = stats.mannwhitneyu(echo_correct, echo_wrong)
    n1, n2 = len(echo_correct), len(echo_wrong)
    rank_biserial = 1 - (2 * u_stat) / (n1 * n2)

    diffs = [
        rng.choice(echo_correct, n1, replace=True).mean() - rng.choice(echo_wrong, n2, replace=True).mean()
        for _ in range(n_boot)
    ]
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])

    out['table12_outcome_conditioned'] = {
        'mean_echo_correct': echo_correct.mean(), 'mean_echo_wrong': echo_wrong.mean(),
        'mannwhitney_p': p_val, 'rank_biserial': rank_biserial,
        'bootstrap_ci': [ci_lo, ci_hi],
    }


def table13_extremity_model(records, out):
    """Continuous logistic model of stated-belief change vs. distance from 0.5."""
    p0 = np.array([r['p0'] for r in records])
    prob_diffs = np.array([r['prob_diff'] for r in records])
    extremity = np.abs(p0 - 0.5)
    changed = (prob_diffs >= EPSILON).astype(int)

    def neg_log_lik(beta):
        b0, b1 = beta
        z = b0 + b1 * extremity
        p = 1 / (1 + np.exp(-z))
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return -np.sum(changed * np.log(p) + (1 - changed) * np.log(1 - p))

    res = minimize(neg_log_lik, [0, 0], method='Nelder-Mead')
    b0, b1 = res.x
    rho, p_rho = stats.spearmanr(extremity, changed)

    out['table13_extremity_model'] = {
        'intercept': b0, 'coefficient': b1, 'spearman_rho': rho, 'spearman_p': p_rho,
    }


def bootstrap_ci_mean(values, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)
    boot_means = [rng.choice(values, len(values), replace=True).mean() for _ in range(n_boot)]
    return np.percentile(boot_means, [2.5, 97.5])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='./data')
    parser.add_argument('--out-dir', default='./results')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out = {}

    peer_traces = load_traces(os.path.join(args.data_dir, 'peer_critique_debate.jsonl'))
    records = build_peer_critique_records(peer_traces)

    echo_vals = np.array([r['echo'] for r in records])
    np.save(os.path.join(args.out_dir, 'bjri_raw_CONFIRMED.npy'), echo_vals)

    table3_ablation(records, out)
    table4_8_crosstab_and_association(records, out)
    table9_10_sensitivity(records, out)
    table11_permutation_test(records, out)
    table12_outcome_conditioned(records, out)
    table13_extremity_model(records, out)

    out['echo_summary'] = {
        'n': len(echo_vals), 'mean': echo_vals.mean(), 'sd': echo_vals.std(),
        'median': float(np.median(echo_vals)),
        'bootstrap_ci': list(bootstrap_ci_mean(echo_vals)),
    }

    # Sequential pipeline: evidence retention
    seq_traces = load_traces(os.path.join(args.data_dir, 'sequential_pipeline.jsonl'))
    retention = []
    for t in seq_traces:
        calls = t['trace']['calls']
        r = compute_evidence_retention(calls[0]['response']['text'], calls[2]['response']['text'])
        if r is not None:
            retention.append(r)
    retention = np.array(retention)
    np.save(os.path.join(args.out_dir, 'retention_raw_CONFIRMED.npy'), retention)
    out['retention_summary'] = {
        'n': len(retention), 'mean': retention.mean(),
        'bootstrap_ci': list(bootstrap_ci_mean(retention)),
    }

    # Orchestrator specialist: evidence incorporation
    orch_traces = load_traces(os.path.join(args.data_dir, 'orchestrator_specialist.jsonl'))
    incorp = []
    for t in orch_traces:
        calls = t['trace']['calls']
        r = compute_evidence_incorporation(calls[1]['response']['text'], calls[2]['response']['text'])
        if r is not None:
            incorp.append(r)
    incorp = np.array(incorp)
    np.save(os.path.join(args.out_dir, 'incorp_raw_FINAL.npy'), incorp)
    out['incorporation_summary'] = {
        'n': len(incorp), 'mean': incorp.mean(), 'median': float(np.median(incorp)),
        'bootstrap_ci': list(bootstrap_ci_mean(incorp)),
    }

    with open(os.path.join(args.out_dir, 'summary_statistics.md'), 'w') as f:
        f.write("# ECHO Analysis — Summary Statistics\n\n")
        f.write("Reproduced by analysis.py. See the manuscript for full table numbering and context.\n\n")
        f.write("```\n")
        f.write(json.dumps(out, indent=2, default=str))
        f.write("\n```\n")

    print(f"Analysis complete. Results written to {args.out_dir}/summary_statistics.md")
    print(json.dumps(out, indent=2, default=str))


if __name__ == '__main__':
    main()
