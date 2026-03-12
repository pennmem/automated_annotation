#!/usr/bin/env python3
"""
Convert a transcription CSV (Word, Onset, Offset, Probability) to a .ann file
matching the LTP annotation format:

    #Begin Header. [Do not edit before this line.]
    #Annotator: <model_name>
    #UTC Locally Formatted: <datetime>
    #UNIX: <unix_timestamp>
    #Program Version: automated_annot
    <blank line>
    onset_ms\titem_num\tword

item_num is the 1-based index of the word in the wordpool (0 if not found or
wordpool not provided). Multi-word pool items (e.g. TRACING_PAPER) are matched
by merging consecutive ASR words before the lookup.

Usage (standalone):
    python csv_to_ann.py whisperx_0.csv whisperx_0.ann --model-name whisperx
    python csv_to_ann.py whisperx_0.csv whisperx_0.ann --wordpool /path/to/wordpool.txt
"""

import os
import sys
import time
import argparse
import pandas as pd
from datetime import datetime, timezone


def _load_wordpool(wordpool_path: str):
    """Return ordered list of uppercased words from a wordpool file."""
    with open(wordpool_path) as f:
        return [line.strip().upper() for line in f if line.strip()]


def _merge_multiword(df: pd.DataFrame, wp_words: list) -> pd.DataFrame:
    """Merge consecutive ASR words that together form a multi-word pool item.

    E.g. if wordpool has TRACING_PAPER and the CSV has TRACING then PAPER
    back-to-back, they become a single TRACING_PAPER row using the onset of
    TRACING and the offset of PAPER.
    """
    multiword = [
        (w.split('_'), w)
        for w in wp_words if '_' in w
    ]
    if not multiword:
        return df

    rows = df.to_dict('records')
    merged = []
    i = 0
    while i < len(rows):
        matched = False
        # Try longest items first to avoid partial matches
        for parts, full_word in sorted(multiword, key=lambda x: -len(x[0])):
            n = len(parts)
            if i + n <= len(rows):
                window = [str(rows[i + j]['Word']).strip().upper() for j in range(n)]
                if window == parts:
                    combined = rows[i].copy()
                    combined['Word'] = full_word
                    combined['Offset'] = rows[i + n - 1].get('Offset', combined.get('Offset'))
                    probs = [rows[i + j].get('Probability') for j in range(n)]
                    probs = [p for p in probs if p is not None and not (isinstance(p, float) and pd.isna(p))]
                    combined['Probability'] = min(probs) if probs else float('nan')
                    merged.append(combined)
                    i += n
                    matched = True
                    break
        if not matched:
            merged.append(rows[i])
            i += 1

    return pd.DataFrame(merged) if merged else df.iloc[0:0].copy()


def csv_to_ann(csv_path: str, ann_path: str, model_name: str = 'automated',
               wordpool: list = None) -> None:
    """Convert a single transcription CSV to .ann format.

    Args:
        csv_path:   Path to input CSV with columns Word, Onset, Offset, Probability.
        ann_path:   Path to write the .ann file.
        model_name: String written into the #Annotator header field.
        wordpool:   Ordered list of uppercased wordpool words. When provided:
                    - Consecutive words matching a multi-word item are merged.
                    - item_num is set to the 1-based wordpool index (0 if not found).
    """
    df = pd.read_csv(csv_path)

    # Drop rows without a valid onset (e.g. Whisper-only backend)
    df = df.dropna(subset=['Onset']).reset_index(drop=True)

    if wordpool:
        # Merge consecutive words into multi-word pool items first
        df = _merge_multiword(df, wordpool)
        # Build 1-based index lookup (underscored form is canonical in pool)
        wp_index = {w: i + 1 for i, w in enumerate(wordpool)}
    else:
        wp_index = {}

    now_unix = int(time.time())
    now_str  = datetime.now(timezone.utc).strftime('%B %d, %Y\t%I:%M:%S %p UTC')

    header_lines = [
        '#Begin Header. [Do not edit before this line. Never edit with an instance of the program open.]',
        f'#Annotator: {model_name}',
        f'#UTC Locally Formatted: {now_str}',
        f'#UNIX: {now_unix}',
        '#Program Version: automated_annot',
        '',   # blank line terminates the header — do not remove
    ]

    data_lines = []
    for _, row in df.iterrows():
        onset = float(row['Onset'])
        word  = str(row['Word']).strip().upper()
        if word.lower() in ('nan', ''):
            continue
        item_num = wp_index.get(word, 0)
        data_lines.append(f'{onset}\t{item_num}\t{word}')

    os.makedirs(os.path.dirname(os.path.abspath(ann_path)), exist_ok=True)
    with open(ann_path, 'w') as f:
        f.write('\n'.join(header_lines + data_lines) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert a transcription CSV to LTP .ann format'
    )
    parser.add_argument('csv_path',      help='Input CSV (Word, Onset, Offset, Probability)')
    parser.add_argument('ann_path',      help='Output .ann file path')
    parser.add_argument('--model-name',  default='automated',
                        help='Written into the #Annotator header field (default: automated)')
    parser.add_argument('--wordpool',    default=None,
                        help='Path to wordpool .txt file (one word per line). '
                             'Enables word-index lookup and multi-word merging.')
    args = parser.parse_args()

    wp = _load_wordpool(args.wordpool) if args.wordpool else None
    csv_to_ann(args.csv_path, args.ann_path, model_name=args.model_name, wordpool=wp)
    print(f'Saved: {args.ann_path}')
