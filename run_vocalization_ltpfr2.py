"""Run VocalizationClassifier + LongDurationVocalization on top-200 ltpFR2 sessions.

Reads base-whisperx CSVs, applies UpperCase → SuffixStripping →
VocalizationClassifier → LongDurationVocalization, writes results to
results/whisperx-claude-ltpfr2/.

Also collects pattern statistics across all sessions for analysis.
"""
import glob
import json
import os
import sys
import traceback
from collections import Counter

import pandas as pd

from postprocessing import (
    LongDurationVocalization,
    SuffixStripping,
    UpperCase,
    VocalizationClassifier,
)

# ── Paths ────────────────────────────────────────────────────────────────────
SUMMARY_CSV = 'intrusion_exploration/ltpfr2_summary_sorted.csv'
BASE_DIR = 'results/base-whisperx/data/eeg/scalp/ltp/ltpFR2'
OUT_ROOT = 'results/whisperx-claude-ltpfr2'
DATA_ROOT = '/data/eeg/scalp/ltp/ltpFR2'
N_SESSIONS = 200

# ── Rules ────────────────────────────────────────────────────────────────────
rules = [
    UpperCase(),
    SuffixStripping(),
    VocalizationClassifier(),
    LongDurationVocalization(),
]


def load_wordpool(subject):
    """Load the wasnorm wordpool for a subject."""
    wp_path = os.path.join(DATA_ROOT, subject, 'wasnorm_wordpool.txt')
    if not os.path.exists(wp_path):
        return None
    with open(wp_path) as f:
        return [line.strip().upper() for line in f if line.strip()]


def load_lst_words(subject, session):
    """Load all .lst files for a session (the presented items)."""
    session_dir = os.path.join(DATA_ROOT, subject, f'session_{session}')
    words = set()
    for lst_file in glob.glob(os.path.join(session_dir, '*.lst')):
        with open(lst_file) as f:
            for line in f:
                w = line.strip().upper()
                if w:
                    words.add(w)
    return words


def process_session(subject, session, patterns):
    """Process all CSVs for one session. Returns (n_files, n_rows)."""
    in_dir = os.path.join(BASE_DIR, subject, f'session_{session}', 'whisperx_out')
    out_dir = os.path.join(OUT_ROOT, 'data', 'eeg', 'scalp', 'ltp', 'ltpFR2',
                           subject, f'session_{session}', 'whisperx_out')

    if not os.path.isdir(in_dir):
        return 0, 0

    os.makedirs(out_dir, exist_ok=True)

    wordpool = load_wordpool(subject)
    if wordpool is None:
        return 0, 0

    lst_words = load_lst_words(subject, session)
    # Merge lst words into wordpool
    wp_set = set(wordpool)
    wordpool_merged = wordpool + [w for w in lst_words if w not in wp_set]

    csv_files = sorted(glob.glob(os.path.join(in_dir, '*.csv')))
    total_rows = 0
    n_files = 0

    for csv_path in csv_files:
        fname = os.path.basename(csv_path)
        list_num_str = fname.replace('.csv', '')
        try:
            list_num = int(list_num_str)
        except ValueError:
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        if df.empty:
            # Copy empty file as-is
            df.to_csv(os.path.join(out_dir, fname), index=False)
            n_files += 1
            continue

        context = {
            'wordpool': wordpool_merged,
            'list_words': lst_words,
            'list_num': list_num,
            'experimenter_overrides': {},
        }

        for rule in rules:
            df = rule.apply(df, context)

        df.to_csv(os.path.join(out_dir, fname), index=False)
        n_files += 1
        total_rows += len(df)

        # Collect pattern stats
        if 'Type' in df.columns:
            for t in df['Type'].values:
                t_str = str(t).strip()
                if t_str and t_str != 'nan':
                    patterns['type_counts'][t_str] += 1
                else:
                    patterns['type_counts']['Guess'] += 1

            # Track intrusion words
            intrusions = df[df['Type'] == 'Intrusion']
            for _, row in intrusions.iterrows():
                w = str(row['Word']).strip().upper()
                patterns['intrusion_words'][w] += 1

            # Track filler count per list
            n_filler = (df['Type'] == 'Filler').sum()
            n_sentence = (df['Type'] == 'Sentence').sum()
            if n_filler > 0 or n_sentence > 0:
                patterns['sessions_with_vocalizations'] += 1

    return n_files, total_rows


def main():
    summary = pd.read_csv(SUMMARY_CSV)
    top = summary.head(N_SESSIONS)

    patterns = {
        'type_counts': Counter(),
        'intrusion_words': Counter(),
        'sessions_with_vocalizations': 0,
        'total_sessions': 0,
        'total_files': 0,
        'total_rows': 0,
    }

    for i, row in top.iterrows():
        subject = row['subject']
        session = int(row['session'])
        patterns['total_sessions'] += 1

        try:
            n_files, n_rows = process_session(subject, session, patterns)
            patterns['total_files'] += n_files
            patterns['total_rows'] += n_rows
            print(f"[{patterns['total_sessions']:3d}/200] {subject} session_{session}: "
                  f"{n_files} files, {n_rows} rows")
        except Exception as e:
            print(f"[{patterns['total_sessions']:3d}/200] ERROR {subject} session_{session}: {e}")
            traceback.print_exc()

    # Write pattern summary
    os.makedirs(OUT_ROOT, exist_ok=True)
    pattern_report = []
    pattern_report.append("=== VOCALIZATION CLASSIFICATION PATTERN REPORT ===")
    pattern_report.append(f"Sessions processed: {patterns['total_sessions']}")
    pattern_report.append(f"Total files: {patterns['total_files']}")
    pattern_report.append(f"Total output rows: {patterns['total_rows']}")
    pattern_report.append(f"Sessions with vocalizations: {patterns['sessions_with_vocalizations']}")
    pattern_report.append("")

    pattern_report.append("--- Type Distribution ---")
    total_events = sum(patterns['type_counts'].values())
    for t, count in patterns['type_counts'].most_common():
        pct = count / total_events * 100 if total_events else 0
        pattern_report.append(f"  {t:20s}: {count:6d} ({pct:.1f}%)")
    pattern_report.append(f"  {'TOTAL':20s}: {total_events:6d}")
    pattern_report.append("")

    pattern_report.append("--- Top 50 Intrusion Words ---")
    for word, count in patterns['intrusion_words'].most_common(50):
        pattern_report.append(f"  {word:20s}: {count:4d}")
    pattern_report.append("")

    report_text = '\n'.join(pattern_report)
    print('\n' + report_text)

    with open(os.path.join(OUT_ROOT, 'pattern_report.txt'), 'w') as f:
        f.write(report_text)

    # Also save raw pattern data as JSON for notebook use
    json_data = {
        'type_counts': dict(patterns['type_counts']),
        'intrusion_words_top100': dict(patterns['intrusion_words'].most_common(100)),
        'total_sessions': patterns['total_sessions'],
        'total_files': patterns['total_files'],
        'total_rows': patterns['total_rows'],
        'sessions_with_vocalizations': patterns['sessions_with_vocalizations'],
    }
    with open(os.path.join(OUT_ROOT, 'pattern_stats.json'), 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"\nDone. Results in {OUT_ROOT}/")


if __name__ == '__main__':
    main()
