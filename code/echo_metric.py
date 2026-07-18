"""
echo_metric.py

The ECHO index (Section 3.3) and the two related evidence-utilization measures
used for the sequential_pipeline and orchestrator_specialist configurations
(Section 3.1, Table 2 of the accompanying manuscript).
"""

from extraction import extract_evidence_terms, extract_probability, jaccard_overlap, STOPLIST_ORCHESTRATOR

EPSILON = 0.005  # threshold below which a stated-probability change is treated as "unchanged"
                 # (justified in Section 3.3: half the smallest reportable increment, 0.01)


def compute_echo(round0_text, roundN_text, final_probability, stoplist=None):
    """
    Compute the ECHO index for a single peer_critique_debate trace.

    ECHO_i = (1 - J(E_0, E_n)) * 1[|delta_p| < epsilon]

    Parameters
    ----------
    round0_text : str
        Response text from the first self-critique round.
    roundN_text : str
        Response text from the final self-critique round.
    final_probability : float
        The trace's reported final probability (from the dataset record, not extracted
        from text, since it is the ground-truth reported value).
    stoplist : set, optional
        Evidence-extraction stoplist to use; defaults to the standard stoplist.

    Returns
    -------
    dict with keys: echo, jaccard, prob_diff, belief_unchanged, p0
    Returns None if no probability could be extracted from round0_text.
    """
    ev0 = extract_evidence_terms(round0_text, stoplist)
    evN = extract_evidence_terms(roundN_text, stoplist)
    jaccard = jaccard_overlap(ev0, evN)

    p0 = extract_probability(round0_text)
    if p0 is None:
        return None

    prob_diff = abs(final_probability - p0)
    belief_unchanged = 1 if prob_diff < EPSILON else 0
    echo = (1 - jaccard) * belief_unchanged

    return {
        'echo': echo,
        'jaccard': jaccard,
        'prob_diff': prob_diff,
        'belief_unchanged': belief_unchanged,
        'p0': p0,
    }


def compute_evidence_retention(upstream_text, downstream_text, stoplist=None):
    """
    Evidence-term retention measure used for sequential_pipeline (Section 3.1):
    the fraction of the upstream stage's evidence terms that also appear in the
    downstream stage's response.

    Returns None if the upstream stage has no extracted evidence terms.
    """
    ev_up = extract_evidence_terms(upstream_text, stoplist)
    ev_down = extract_evidence_terms(downstream_text, stoplist)
    if not ev_up:
        return None
    return len(ev_up & ev_down) / len(ev_up)


def compute_evidence_incorporation(specialist_text, integrator_text, stoplist=None):
    """
    Evidence-term incorporation measure used for orchestrator_specialist (Section 3.1):
    the fraction of the specialist's evidence terms that also appear in the
    integrator's final response. Uses the extended orchestrator stoplist by default,
    since role names (Specialist, Integrator, Planner) would otherwise be
    misclassified as evidence terms in this configuration.

    Returns None if the specialist stage has no extracted evidence terms.
    """
    if stoplist is None:
        stoplist = STOPLIST_ORCHESTRATOR
    ev_spec = extract_evidence_terms(specialist_text, stoplist)
    ev_integ = extract_evidence_terms(integrator_text, stoplist)
    if not ev_spec:
        return None
    return len(ev_spec & ev_integ) / len(ev_spec)
