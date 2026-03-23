"""Process annotation gauntlet CSVs using VocalizationClassifier from postprocessing.

This script applies the postprocessing OutputRule pipeline to the annotation
gauntlet data, demonstrating how VocalizationClassifier integrates with the
existing rule system.
"""
import os
import pandas as pd
from postprocessing import (
    VocalizationClassifier,
    LongDurationVocalization,
    UpperCase,
    SuffixStripping,
)

IN_DIR = 'results/base-whisperx/annotation_gauntlet/session_0/whisperx_out'
OUT_DIR = 'results/whisperx-claude-focused-1/annotation_gauntlet/session_0/whisperx_out'
LST_DIR = 'dependencies/annotation_gauntlet/session_0'

os.makedirs(OUT_DIR, exist_ok=True)

# Load wordpool
with open('dependencies/annotation_gauntlet/RAM_wordpool.txt') as f:
    WORDPOOL = [line.strip().upper() for line in f if line.strip()]

# Manual experimenter overrides for rows that can't be auto-detected
# (e.g. due to gap splitting between experimenter speech fragments)
EXPERIMENTER_OVERRIDES = {
    0: set(range(1, 19)),   # "IS THIS... RECALL ALL THE WORDS YOU CAN REMEMBER"
    7: set(range(5, 11)),   # "COME IN ITS OKAY COME BACK"
}

# Build rules
rules = [
    UpperCase(),
    SuffixStripping(),
    VocalizationClassifier(),
    LongDurationVocalization(),
]


def load_lst(list_num):
    path = os.path.join(LST_DIR, f'{list_num}.lst')
    with open(path) as f:
        return set(line.strip().upper() for line in f if line.strip())


def process_csv(list_num):
    df = pd.read_csv(os.path.join(IN_DIR, f'{list_num}.csv'))
    lst_words = load_lst(list_num)

    # Build context matching the pipeline convention
    context = {
        'wordpool': WORDPOOL + [w for w in lst_words if w not in set(WORDPOOL)],
        'list_words': lst_words,
        'list_num': list_num,
        'experimenter_overrides': EXPERIMENTER_OVERRIDES,
    }

    # Apply rules sequentially
    for rule in rules:
        df = rule.apply(df, context)

    df.to_csv(os.path.join(OUT_DIR, f'{list_num}.csv'), index=False)

    # Build reasoning log
    reasoning = [f"=== List {list_num} ==="]
    reasoning.append(f"List words: {sorted(lst_words)}")
    for _, row in df.iterrows():
        w = str(row['Word']).strip()
        t = str(row.get('Type', '')).strip()
        if t == '':
            reasoning.append(f"  {w} @ {int(row['Onset'])}ms -> Wordpool guess")
        elif t == 'Extension':
            reasoning.append(f"  <> @ {int(row['Onset'])}ms -> Extension")
        else:
            reasoning.append(f"  {w} @ {int(row['Onset'])}ms -> {t}")
    reasoning.append(f"Output rows: {len(df)}")

    with open(os.path.join(OUT_DIR, f'{list_num}_reasoning.txt'), 'w') as f:
        f.write('\n'.join(reasoning))

    return df


for i in range(12):
    out = process_csv(i)
    print(f"\n=== List {i} ({len(out)} rows) ===")
    print(out.to_string(index=False))
