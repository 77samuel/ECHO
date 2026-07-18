"""
extraction.py

Evidence-term extraction and stated-probability extraction from ForesightFlow
response text, as described in Section 3.2 of the accompanying manuscript.
"""

import re

# Stoplist of ~40 generic discourse words that would otherwise be misclassified
# as evidence terms by the capitalized-word heuristic below.
STOPLIST = {
    'This', 'However', 'Assessment', 'Factors', 'Let', 'General', 'Given', 'Based',
    'Key', 'Considerations', 'Despite', 'Overall', 'Final', 'Summary', 'Conclusion',
    'Additionally', 'Therefore', 'Furthermore', 'Moreover', 'While', 'Since', 'After',
    'Current', 'Market', 'Question', 'Resolution', 'Answer', 'My', 'Your', 'The', 'In',
    'For', 'On', 'At', 'As', 'If', 'It', 'I', 'We', 'They', 'He', 'She', 'Search', 'Results',
    'Note', 'Important', 'Analysis', 'Estimate', 'Probability', 'Reasoning', 'Response',
    'Historical', 'Buying', 'Launch', 'There', 'Combined', 'Risk', 'Slight', 'Without',
    'Reassessment', 'Capital', 'Occasional', 'Inability'
}

# Extended stoplist used specifically for the orchestrator_specialist configuration,
# where role names themselves (Specialist, Integrator, Planner) would otherwise be
# misclassified as evidence terms.
STOPLIST_ORCHESTRATOR = STOPLIST | {
    'Web', 'Direction', 'Strong', 'Decision', 'Condition',
    'Specialist', 'Integrator', 'Planner', 'Report'
}


def extract_evidence_terms(text, stoplist=None):
    """
    Extract a set of candidate evidence terms from a response string.

    Captures, in order:
      - multi-word capitalized phrases (proper nouns / organizations)
      - single capitalized words that are NOT sentence-initial
      - percentages (e.g. "42%")
      - monetary amounts (e.g. "$4B", "$1,200")
      - four-digit years
      - specific calendar dates (e.g. "October 25")

    All candidates are filtered against `stoplist` (defaults to STOPLIST) to remove
    common discourse words that would otherwise be misclassified as evidence.
    """
    if stoplist is None:
        stoplist = STOPLIST

    terms = set()

    # Multi-word capitalized phrases
    multiword = re.findall(r'\b[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,}){1,3}\b', text)
    for m in multiword:
        if not any(w in stoplist for w in m.split()):
            terms.add(m)

    # Single capitalized words, excluding sentence-initial position
    for m in re.finditer(r'(?<![\.\!\?]\s)(?<!^)\b[A-Z][a-zA-Z]{3,}\b', text):
        w = m.group()
        if w not in stoplist and len(w) > 3:
            terms.add(w)

    # Numeric evidence: percentages and dollar amounts.
    # NOTE: an earlier draft of this function also matched four-digit years and
    # specific calendar dates (e.g. "October 25"). Those two patterns are NOT
    # included here, because the analysis reported in the manuscript was run
    # without them; adding them changes the extracted term sets enough to shift
    # the reported summary statistics (verified during repository preparation).
    terms |= set(re.findall(r'\b\d{1,3}(?:\.\d+)?%\b', text))
    terms |= set(re.findall(r'\$\d[\d,]*\.?\d*[BMK]?\b', text))

    return terms


def extract_probability(text):
    """
    Extract the first stated probability value (format 0.XX) from response text.
    Returns None if no such value is found.
    """
    nums = re.findall(r'\b0\.\d{1,2}\b', text)
    return float(nums[0]) if nums else None


def jaccard_overlap(set_a, set_b):
    """
    Jaccard similarity between two sets of extracted evidence terms.
    Returns 1.0 if both sets are empty (defined as fully overlapping / no evidence to compare).
    """
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)
