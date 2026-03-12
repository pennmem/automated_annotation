#!/usr/bin/env python3
"""
Cron-compatible automated annotation script for LTP sessions.

Workflow:
  1. Read active experiments from /data/eeg/scalp/ltp/ACTIVE_EXPERIMENTS.txt
  2. Find session directories whose {exp}.json was modified within --max-age-days
  3. Skip sessions that already have a *.ann file (already human-annotated)
  4. Run the chosen transcription backend on each unannotated session in parallel
  5. Save per-trial CSVs and .ann files alongside existing session data:
       {session_dir}/{model_name}_{trial_num}.csv
       {session_dir}/{model_name}_{trial_num}.ann

Crontab example (run every hour):
    0 * * * * /path/to/conda/envs/annotate_np1/bin/python \\
        /home1/zrentala/automated_annotation/run_auto_annot_cron.py \\
        --backend whisperx --use-gpu >> /home1/zrentala/automated_annotation/logs/cron.log 2>&1
"""

import os
import sys
import glob
import time
import shutil
import logging
import argparse
import tempfile
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

LTP_ROOT                = '/data/eeg/scalp/ltp'
ACTIVE_EXPERIMENTS_FILE = os.path.join(LTP_ROOT, 'ACTIVE_EXPERIMENTS.txt')
DEFAULT_MAX_AGE_DAYS    = 7
DEFAULT_BACKEND         = 'whisperx'

os.makedirs(os.path.join(SCRIPT_DIR, 'logs'), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(SCRIPT_DIR, 'logs', 'annotation_cron.log')),
    ]
)
logger = logging.getLogger(__name__)


# ── Experiment / session discovery ────────────────────────────────────────

def load_active_experiments(path: str = ACTIVE_EXPERIMENTS_FILE):
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith('#')]


def find_sessions_needing_annotation(experiments, max_age_days: float = DEFAULT_MAX_AGE_DAYS):
    """Return session dirs that have a recently modified {exp}.json but no .ann file."""
    cutoff = time.time() - max_age_days * 86400
    sessions = []

    for exp in experiments:
        exp_dir = os.path.join(LTP_ROOT, exp)
        if not os.path.isdir(exp_dir):
            logger.warning(f'Experiment directory not found: {exp_dir}')
            continue

        pattern = os.path.join(exp_dir, '*', 'session_*', f'{exp}.json')
        for json_path in glob.glob(pattern):
            if os.path.getmtime(json_path) < cutoff:
                continue

            session_dir = os.path.dirname(json_path)

            if glob.glob(os.path.join(session_dir, '*.ann')):
                logger.debug(f'Skipping (already annotated): {session_dir}')
                continue

            if not _numbered_wavs(session_dir):
                logger.debug(f'Skipping (no numbered wav files): {session_dir}')
                continue

            if not os.access(session_dir, os.W_OK):
                logger.warning(f'Skipping (no write permission): {session_dir}')
                continue

            sessions.append(session_dir)
            logger.info(f'Found session needing annotation: {session_dir}')

    return sessions


def _numbered_wavs(session_dir: str):
    return sorted(
        f for f in glob.glob(os.path.join(session_dir, '*.wav'))
        if os.path.basename(f)[0].isdigit()
    )


def _available_gpu_devices():
    """Return list of available CUDA device strings, e.g. ['cuda:0', 'cuda:1']."""
    try:
        import torch
        if torch.cuda.is_available():
            return [f'cuda:{i}' for i in range(torch.cuda.device_count())]
    except ImportError:
        pass
    return []


# ── Per-session worker (must be top-level for ProcessPoolExecutor pickling) ──

def _annotate_worker(session_dir: str, backend_name: str, model_name: str,
                     use_gpu: bool, device: str, force_recompute: bool) -> str:
    """Run in a subprocess: transcribe one session and write CSV + .ann files.

    Returns session_dir on success, raises on failure.
    """
    # Re-insert path in the worker process
    import sys, os, glob, shutil, tempfile, logging
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    log = logging.getLogger(__name__)

    from automated_annot import run_transcription
    from backends import get_backend
    from csv_to_ann import csv_to_ann

    wav_files = sorted(
        f for f in glob.glob(os.path.join(session_dir, '*.wav'))
        if os.path.basename(f)[0].isdigit()
    )
    log.info(f'[{device or "cpu"}] Annotating {len(wav_files)} trial(s) in {session_dir}')

    with tempfile.TemporaryDirectory() as tmp:
        run_transcription(
            session_dir, tmp,
            backend_name=backend_name,
            use_gpu=use_gpu,
            device=device,
            force_recompute=force_recompute,
        )

        backend    = get_backend(backend_name)
        out_subdir = os.path.join(tmp, backend.output_subdir)
        csv_files  = sorted(glob.glob(os.path.join(out_subdir, '*.csv')))

        if not csv_files:
            raise RuntimeError(f'No CSVs produced for {session_dir}')

        for csv_path in csv_files:
            trial_num = os.path.splitext(os.path.basename(csv_path))[0]
            dest_csv  = os.path.join(session_dir, f'{model_name}_{trial_num}.csv')
            dest_ann  = os.path.join(session_dir, f'{model_name}_{trial_num}.ann')

            try:
                shutil.copy2(csv_path, dest_csv)
            except PermissionError:
                raise PermissionError(
                    f'No write permission for {dest_csv} — '
                    f'check that your user owns {session_dir}'
                )

            log.info(f'  Saved CSV: {dest_csv}')
            csv_to_ann(dest_csv, dest_ann, model_name=model_name)
            log.info(f'  Saved ANN: {dest_ann}')

    return session_dir


# ── Entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Automated annotation cron job for LTP sessions',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--backend', default=DEFAULT_BACKEND,
                        choices=['whisperx', 'whisper', 'assemblyai'])
    parser.add_argument('--model-name', default=None,
                        help='Name used in output filenames and #Annotator field. '
                             'Defaults to the backend name.')
    parser.add_argument('--max-age-days', type=float, default=DEFAULT_MAX_AGE_DAYS,
                        help='Only consider sessions modified within this many days')
    parser.add_argument('--use-gpu', action='store_true',
                        help='Enable GPU acceleration (whisperx/whisper only)')
    parser.add_argument('--device', default=None,
                        help='Fix all workers to one device, e.g. cuda:0. '
                             'By default each worker gets its own GPU round-robin.')
    parser.add_argument('--workers', type=int, default=None,
                        help='Parallel worker processes. Defaults to number of GPUs '
                             'when --use-gpu, else 4.')
    parser.add_argument('--no-parallel', action='store_true',
                        help='Disable parallelism and run sessions sequentially.')
    parser.add_argument('--force-recompute', action='store_true')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print sessions that would be annotated, then exit')
    parser.add_argument('--experiments-file', default=ACTIVE_EXPERIMENTS_FILE)
    args = parser.parse_args()

    model_name  = args.model_name or args.backend
    experiments = load_active_experiments(args.experiments_file)
    logger.info(f'Active experiments: {experiments}')

    sessions = find_sessions_needing_annotation(
        experiments, max_age_days=args.max_age_days
    )
    logger.info(f'Sessions needing annotation: {len(sessions)}')

    if args.dry_run:
        for s in sessions:
            print(s)
        return

    if not sessions:
        return

    # ── Device assignment ──────────────────────────────────────────────────
    # If a fixed --device is given, every worker uses it.
    # Otherwise with --use-gpu, round-robin across all visible GPUs.
    if args.device:
        devices = [args.device] * len(sessions)
    elif args.use_gpu:
        gpus = _available_gpu_devices()
        if not gpus:
            logger.warning('--use-gpu set but no CUDA devices found; falling back to CPU')
            devices = [None] * len(sessions)
        else:
            devices = [gpus[i % len(gpus)] for i in range(len(sessions))]
            logger.info(f'Distributing {len(sessions)} sessions across {len(gpus)} GPU(s): {gpus}')
    else:
        devices = [None] * len(sessions)

    # ── Worker count ───────────────────────────────────────────────────────
    # GPU workloads: default to 1 worker per GPU, capped at number of GPUs.
    # Running multiple workers per GPU causes OOM when each loads a full model.
    if args.no_parallel:
        n_workers = 1
    elif args.workers is not None:
        n_workers = args.workers
    elif args.use_gpu:
        n_gpus = len(_available_gpu_devices())
        # One worker per GPU; if only 1 GPU (or none), run sequentially in-process
        n_workers = n_gpus if n_gpus > 1 else 1
    else:
        n_workers = min(len(sessions), 4)

    logger.info(f'Running with {n_workers} parallel worker(s)')

    # ── Dispatch ───────────────────────────────────────────────────────────
    job_args = [
        (session_dir, args.backend, model_name, args.use_gpu, device, args.force_recompute)
        for session_dir, device in zip(sessions, devices)
    ]

    if n_workers == 1:
        for jargs in job_args:
            try:
                _annotate_worker(*jargs)
            except Exception:
                logger.exception(f'Failed to annotate {jargs[0]}')
    else:
        mp_ctx = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_ctx) as pool:
            futures = {pool.submit(_annotate_worker, *jargs): jargs[0] for jargs in job_args}
            for future in as_completed(futures):
                session_dir = futures[future]
                try:
                    future.result()
                    logger.info(f'Finished: {session_dir}')
                except Exception:
                    logger.exception(f'Failed to annotate {session_dir}')


if __name__ == '__main__':
    main()
