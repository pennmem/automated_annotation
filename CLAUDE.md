# Automated Annotation

Automated speech annotation of LTP (Long-Term Potentiation) free-recall recordings using ASR models with word-level timing.

## Architecture

```
run_automated_annotations.ipynb   # Data discovery, train/val/test splits, directory mapping
        │
        ▼
run_automated_annotations.py      # Batch runner: loads pickled dirs, dispatches via Dask/SLURM or sequential
        │
        ▼
automated_annot.py                # Per-session orchestration: WAV discovery, checkpointing, backend dispatch
        │
        ▼
backends.py                       # Backend implementations (WhisperX, Whisper, AssemblyAI)
        │
        ▼
analyze_annot_performance.py      # Post-hoc evaluation: WER, onset diffs, phoneme/confidence analysis
```

## Pipeline Flow

1. **Notebook** loads protocol index from `/protocols/ltp.json`, discovers `.wav` files on disk under `/data/eeg/scalp/ltp/{experiment}/{subject}/session_{N}/`, builds train/val/test split (50/20/30) grouped by subject+experiment, and saves pickled directory mappings (`input_dirs.pkl`, `output_dirs.pkl`).

2. **`run_automated_annotations.py`** loads the pickled mappings, selects a backend based on the `--tag` argument (e.g. `base-whisperx`), and dispatches sessions via Dask/SLURM (`--use-dask`) or runs sequentially.

3. **`automated_annot.py:run_transcription()`** handles per-session processing: discovers `.wav` files, skips files with existing outputs (checkpointing), loads the model once per session, and calls the backend for each file.

4. **`backends.py`** provides three backends via `TranscriptionBackend` ABC:
   - **WhisperXBackend** (`whisperx_out/`): Whisper large-v2 + wav2vec2 forced alignment → word-level onset/offset in ms
   - **WhisperBackend** (`whisper_out/`): Whisper transcription only, no alignment (NaN timing)
   - **AssemblyAIBackend** (`assemblyai_out/`): Cloud API with word timestamps

## Output Format

Each `.wav` produces a `.csv` with columns: `Word, Onset, Offset, Probability`

```
results/{tag}/{split}/data/eeg/scalp/ltp/{experiment}/{subject}/session_{N}/
└── whisperx_out/
    ├── 0.csv
    ├── 1.csv
    └── ...
```

## Key CLI Arguments

### run_automated_annotations.py
- `--tag`: Run identifier (e.g. `base-whisperx`); selects backend by substring match
- `--use-dask`: Distribute via SLURM/Dask (9GB/job, max 150 parallel)
- `--use-gpu`: Enable GPU acceleration
- `--device`: CUDA device string (default: `cuda:0`). Only used when `--use-gpu` is set
- `--smokescreen`: Test mode (1 session)
- `--force_recompute`: Skip checkpointing, reprocess everything

### automated_annot.py (direct usage)
- `--input_dir`, `--output_dir`: Session directories
- `--backend`: `whisper`, `whisperx`, or `assemblyai`
- `--use_gpu`, `--device`, `--smokescreen`, `--force_recompute`

## Checkpointing

Two levels:
1. **Session-level** (`run_automated_annotations.py`): Skips entire sessions where all expected output CSVs exist
2. **File-level** (`automated_annot.py`): Skips individual `.wav` files whose `.csv` output already exists

Both are bypassed with `--force_recompute`.

## Dependencies

Core: `whisperx`, `faster-whisper`, `transformers`, `torch`, `torchaudio`, `assemblyai`
Analysis: `jiwer`, `eng_to_ipa`, `scipy`, `pandas`, `numpy`
Cluster: `cmldask`, `dask.distributed`
