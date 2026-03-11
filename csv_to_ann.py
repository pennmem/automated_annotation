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

item_num is set to 0 for automated annotations (word pool index is unknown at
inference time). The blank line after the last # comment is required — it marks
the end of the header that the LTP annotation tools use to skip metadata.

Usage (standalone):
    python csv_to_ann.py whisperx_0.csv whisperx_0.ann --model-name whisperx
"""

import os
import sys
import time
import argparse
import pandas as pd
from datetime import datetime, timezone


def csv_to_ann(csv_path: str, ann_path: str, model_name: str = 'automated') -> None:
    """Convert a single transcription CSV to .ann format.

    Args:
        csv_path:   Path to input CSV with columns Word, Onset, Offset, Probability.
        ann_path:   Path to write the .ann file.
        model_name: String written into the #Annotator header field and used
                    for output filenames by the cron script.
    """
    df = pd.read_csv(csv_path)

    # Drop rows without a valid onset (e.g. Whisper-only backend)
    df = df.dropna(subset=['Onset']).reset_index(drop=True)

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
        onset    = float(row['Onset'])
        item_num = 0           # word-pool index unknown for automated annotations
        word     = str(row['Word']).strip()
        # Preserve vocalization markers
        if word.lower() in ('nan', '') :
            continue
        data_lines.append(f'{onset}\t{item_num}\t{word}')

    os.makedirs(os.path.dirname(os.path.abspath(ann_path)), exist_ok=True)
    with open(ann_path, 'w') as f:
        f.write('\n'.join(header_lines + data_lines) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert a transcription CSV to LTP .ann format'
    )
    parser.add_argument('csv_path',     help='Input CSV (Word, Onset, Offset, Probability)')
    parser.add_argument('ann_path',     help='Output .ann file path')
    parser.add_argument('--model-name', default='automated',
                        help='Written into the #Annotator header field (default: automated)')
    args = parser.parse_args()

    csv_to_ann(args.csv_path, args.ann_path, model_name=args.model_name)
    print(f'Saved: {args.ann_path}')
