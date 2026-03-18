#!/usr/bin/env python3
"""
Format converters for LTP annotation files.

Supported formats
-----------------
- **.csv** – transcription output: ``Word, Onset, Offset, Probability``
- **.ann** – LTP annotation: header block ending with blank line, then
  tab-separated ``onset_ms  item_num  word``
- **.par** – parsed annotation: tab-separated ``onset_ms  item_num  word``
  (no header, vocalisations written as ``VV``)

Functions
---------
- csv_to_ann  – transcription CSV  → .ann
- ann_to_csv  – .ann               → transcription CSV
- ann_to_par  – .ann               → .par
- par_to_ann  – .par               → .ann

CLI usage::

    python converters.py csv_to_ann  in.csv  out.ann  [--model-name whisperx] [--wordpool wp.txt]
    python converters.py ann_to_csv  in.ann  out.csv
    python converters.py ann_to_par  in.ann  out.par
    python converters.py par_to_ann  in.par  out.ann  [--model-name annotator]
"""

import os
import time
import argparse
import pandas as pd
from datetime import datetime, timezone


# ─── helpers ──────────────────────────────────────────────────────────────────

def _load_wordpool(wordpool_path: str):
    """Return ordered list of uppercased words from a wordpool file."""
    with open(wordpool_path) as f:
        return [line.strip().upper() for line in f if line.strip()]


def _ann_header(model_name: str = 'automated') -> list[str]:
    """Build standard LTP .ann header lines (including trailing blank line)."""
    now_unix = int(time.time())
    now_str = datetime.now(timezone.utc).strftime('%B %d, %Y\t%I:%M:%S %p UTC')
    return [
        '#Begin Header. [Do not edit before this line. Never edit with an instance of the program open.]',
        f'#Annotator: {model_name}',
        f'#UTC Locally Formatted: {now_str}',
        f'#UNIX: {now_unix}',
        '#Program Version: automated_annot',
        '',  # blank line terminates header
    ]


def _find_header_end(filepath: str) -> int:
    """Return the 1-based line number of the first blank line (header end)."""
    with open(filepath) as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                return lineno
    return 0


def load_ann(ann_path: str) -> pd.DataFrame:
    """Read a .ann file into a DataFrame with columns onset, item_num, item_name."""
    skip = _find_header_end(ann_path)
    df = pd.read_csv(ann_path, delimiter='\t',
                     names=['onset', 'item_num', 'item_name'],
                     skiprows=skip, comment='#')
    df['onset'] = pd.to_numeric(df['onset'], errors='coerce')
    df = df.dropna(subset=['onset'])
    return df


def load_par(par_path: str) -> pd.DataFrame:
    """Read a .par file into a DataFrame with columns onset, item_num, item_name."""
    df = pd.read_csv(par_path, delimiter='\t',
                     names=['onset', 'item_num', 'item_name'],
                     comment='#')
    df['onset'] = pd.to_numeric(df['onset'], errors='coerce')
    df = df.dropna(subset=['onset'])
    return df


# ─── converters ───────────────────────────────────────────────────────────────

def csv_to_ann(csv_path: str, ann_path: str, model_name: str = 'automated',
               wordpool: list = None) -> None:
    """Convert a transcription CSV (Word, Onset, Offset, Probability) to .ann."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['Onset']).reset_index(drop=True)

    header = _ann_header(model_name)

    data_lines = []
    for _, row in df.iterrows():
        onset = float(row['Onset'])
        word = str(row['Word']).strip().upper()
        if word.lower() in ('nan', ''):
            continue
        item_num = int(row['item_num']) if pd.notna(row.get('item_num')) else 0
        data_lines.append(f'{onset}\t{item_num}\t{word}')

    os.makedirs(os.path.dirname(os.path.abspath(ann_path)), exist_ok=True)
    with open(ann_path, 'w') as f:
        f.write('\n'.join(header + data_lines) + '\n')


def ann_to_csv(ann_path: str, csv_path: str) -> None:
    """Convert a .ann file to transcription CSV (Word, Onset, Offset, Probability).

    Offset and Probability are set to NaN since .ann files don't store them.
    """
    df = load_ann(ann_path)

    out = pd.DataFrame({
        'Word': df['item_name'],
        'Onset': df['onset'],
        'Offset': float('nan'),
        'Probability': float('nan'),
    })

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    out.to_csv(csv_path, index=False)


def ann_to_par(ann_path: str, par_path: str) -> None:
    """Convert a .ann file to .par format.

    Replicates the logic of the original ann2par.py:
    - Onset is rounded to the nearest integer.
    - Words starting with '<' (vocalisations) are replaced with 'VV'.
    """
    df = load_ann(ann_path)

    os.makedirs(os.path.dirname(os.path.abspath(par_path)), exist_ok=True)
    with open(par_path, 'w') as f:
        for _, row in df.iterrows():
            onset = int(round(float(row['onset'])))
            item_num = int(row['item_num'])
            word = str(row['item_name']).strip()
            if word.startswith('<'):
                word = 'VV'
            f.write(f'{onset}\t{item_num}\t{word}\n')


def par_to_ann(par_path: str, ann_path: str, model_name: str = 'automated') -> None:
    """Convert a .par file to .ann format.

    Onset values are kept as-is (integer ms). A standard .ann header is written.
    """
    df = load_par(par_path)

    header = _ann_header(model_name)

    data_lines = []
    for _, row in df.iterrows():
        onset = float(row['onset'])
        item_num = int(row['item_num'])
        word = str(row['item_name']).strip()
        data_lines.append(f'{onset}\t{item_num}\t{word}')

    os.makedirs(os.path.dirname(os.path.abspath(ann_path)), exist_ok=True)
    with open(ann_path, 'w') as f:
        f.write('\n'.join(header + data_lines) + '\n')


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert between LTP annotation formats (.csv, .ann, .par)')
    sub = parser.add_subparsers(dest='command', required=True)

    # csv_to_ann
    p = sub.add_parser('csv_to_ann', help='Transcription CSV → .ann')
    p.add_argument('input', help='Input .csv file')
    p.add_argument('output', help='Output .ann file')
    p.add_argument('--model-name', default='automated')
    p.add_argument('--wordpool', default=None,
                   help='Wordpool .txt for item_num lookup')

    # ann_to_csv
    p = sub.add_parser('ann_to_csv', help='.ann → transcription CSV')
    p.add_argument('input', help='Input .ann file')
    p.add_argument('output', help='Output .csv file')

    # ann_to_par
    p = sub.add_parser('ann_to_par', help='.ann → .par')
    p.add_argument('input', help='Input .ann file')
    p.add_argument('output', help='Output .par file')

    # par_to_ann
    p = sub.add_parser('par_to_ann', help='.par → .ann')
    p.add_argument('input', help='Input .par file')
    p.add_argument('output', help='Output .ann file')
    p.add_argument('--model-name', default='automated')

    args = parser.parse_args()

    if args.command == 'csv_to_ann':
        wp = _load_wordpool(args.wordpool) if args.wordpool else None
        csv_to_ann(args.input, args.output, model_name=args.model_name, wordpool=wp)
    elif args.command == 'ann_to_csv':
        ann_to_csv(args.input, args.output)
    elif args.command == 'ann_to_par':
        ann_to_par(args.input, args.output)
    elif args.command == 'par_to_ann':
        par_to_ann(args.input, args.output)

    print(f'Saved: {args.output}')
