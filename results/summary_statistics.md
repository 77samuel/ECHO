# ECHO Analysis — Summary Statistics

Reproduced by analysis.py. See the manuscript for full table numbering and context.

```
{
  "table3_ablation": {
    "filtered_evidence_dissimilarity": {
      "r": -0.1263771026777383,
      "p": 0.21497687727265552
    },
    "unfiltered_text_dissimilarity": {
      "r": -0.16375080129868386,
      "p": 0.1071477733963653
    }
  },
  "table4_crosstab": {
    "a": 18,
    "b": 40,
    "c": 9,
    "d": 31
  },
  "table8_association": {
    "odds_ratio": 1.55,
    "fisher_p": 0.49060875070266974,
    "phi": 0.0938880115766874,
    "spearman_rho": 0.11670015630439425,
    "spearman_p": 0.25247235373657845
  },
  "table9_jaccard_sensitivity": {
    "0.2": {
      "odds_ratio": 2.0701754385964914,
      "p": 0.17236714701671405
    },
    "0.25": {
      "odds_ratio": 1.5666666666666667,
      "p": 0.35567305877179317
    },
    "0.3": {
      "odds_ratio": 1.4393939393939394,
      "p": 0.5000635179513155
    },
    "0.35": {
      "odds_ratio": 1.55,
      "p": 0.49060875070266974
    },
    "0.4": {
      "odds_ratio": 1.5714285714285714,
      "p": 0.46047259457148176
    },
    "0.45": {
      "odds_ratio": 1.018181818181818,
      "p": 1.0
    }
  },
  "table10_epsilon_sensitivity": {
    "0": 72.44897959183673,
    "0.005": 72.44897959183673,
    "0.01": 77.55102040816327,
    "0.02": 82.6530612244898,
    "0.05": 92.85714285714286
  },
  "table11_permutation": {
    "observed_rho": 0.11670015630439425,
    "null_mean": 0.002137646480957772,
    "null_sd": 0.10274777910879879,
    "p_value": 0.25
  },
  "table12_outcome_conditioned": {
    "mean_echo_correct": 0.4721444585613621,
    "mean_echo_wrong": 0.4641748132552749,
    "mannwhitney_p": 0.7542767911178028,
    "rank_biserial": -0.03678929765886285,
    "bootstrap_ci": [
      -0.1166785100247361,
      0.13851789770912606
    ]
  },
  "table13_extremity_model": {
    "intercept": -0.3369938097408924,
    "coefficient": -2.7442423734974937,
    "spearman_rho": -0.1841622870279642,
    "spearman_p": 0.06948057964651884
  },
  "echo_summary": {
    "n": 98,
    "mean": 0.46840360464217834,
    "sd": 0.321945931460632,
    "median": 0.5941176470588235,
    "bootstrap_ci": [
      0.4034129665846179,
      0.5325827205301549
    ]
  },
  "retention_summary": {
    "n": 98,
    "mean": 0.13730959623283204,
    "bootstrap_ci": [
      0.12238970128554373,
      0.15316553118773107
    ]
  },
  "incorporation_summary": {
    "n": 98,
    "mean": 0.31806739107797477,
    "median": 0.2916666666666667,
    "bootstrap_ci": [
      0.28977703710147107,
      0.34795268160659704
    ]
  }
}
```
