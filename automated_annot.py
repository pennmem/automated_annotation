import os
import sys
import string
import pandas as pd
import json
import argparse
import warnings

from backends import get_backend


# ─── Whisper-specific helpers (kept for external callers) ───

def get_model_dictionary(language_code='en', model_name=None, model_dir=None):
    # adapted from https://github.com/m-bain/whisperX/blob/main/whisperx/alignment.py
    import torchaudio
    from whisperx.alignment import DEFAULT_ALIGN_MODELS_TORCH, DEFAULT_ALIGN_MODELS_HF

    if model_name is None:
        if language_code in DEFAULT_ALIGN_MODELS_TORCH:
            model_name = DEFAULT_ALIGN_MODELS_TORCH[language_code]
        elif language_code in DEFAULT_ALIGN_MODELS_HF:
            model_name = DEFAULT_ALIGN_MODELS_HF[language_code]
        else:
            print(f"There is no default alignment model set for this language ({language_code}).\
                Please find a wav2vec2.0 model finetuned on this language in https://huggingface.co/models, then pass the model name in --align_model [MODEL_NAME]")
            raise ValueError(f"No default align-model for language: {language_code}")

    if model_name in torchaudio.pipelines.__all__:
        bundle = torchaudio.pipelines.__dict__[model_name]
        labels = bundle.get_labels()
        align_dictionary = {c.lower(): i for i, c in enumerate(labels)}

    return align_dictionary


def clean_whisper_transcript(transcript, model_dictionary=None, model_lang='en'):
    # adapted from https://github.com/m-bain/whisperX/blob/main/whisperx/alignment.py
    from whisperx.alignment import LANGUAGES_WITHOUT_SPACES

    for sdx, segment in enumerate(transcript):
        text = segment["text"].strip()

        if model_lang not in LANGUAGES_WITHOUT_SPACES:
            per_word = text.split(" ")
        else:
            per_word = text

        clean_wrd = []
        for wdx, wrd in enumerate(per_word):
            if isinstance(model_dictionary, type(None)):
                clean_wrd.append(wrd)
            else:
                if any([c in model_dictionary.keys() for c in wrd]):
                    clean_wrd.append(wrd)
    return clean_wrd


# ─── Generic transcription runner ───

def run_transcription(in_dir, out_dir, backend_name, args=dict(),
                      use_gpu=False, smokescreen=False, force_recompute=False, verbose=False):
    """Generic transcription runner that delegates to a backend.

    Handles all shared boilerplate: input validation, output dir creation,
    WAV file discovery, skip-if-done, smokescreen limiting.
    """
    if verbose:
        print(f'starting run_transcription ({backend_name}) with')
        print(in_dir)
        print(out_dir)

    if not os.path.exists(in_dir):
        print(f"The input directory {in_dir} does not exist.")
        sys.exit(1)

    if not os.path.isdir(in_dir):
        print(f"The input path {in_dir} is not a directory.")
        sys.exit(1)

    backend = get_backend(backend_name)

    out_dir = os.path.join(out_dir, backend.output_subdir)
    os.makedirs(out_dir, exist_ok=True)

    wav_files = [f for f in os.listdir(in_dir) if f.endswith('.wav')]

    if wav_files and not force_recompute:
        tmp = []
        for file in wav_files:
            output_name = os.path.splitext(file)[0] + ".csv"
            savepath = os.path.join(out_dir, output_name)
            if os.path.exists(savepath):
                print(f'Already produced output {savepath}. Skipping {file}.')
                continue
            tmp.append(file)
        wav_files = tmp

    if smokescreen:
        wav_files = wav_files[:1]

    if wav_files:
        backend.load_model(use_gpu=use_gpu, smokescreen=smokescreen, args=args)

        for file in wav_files:
            print(f"\n\nProcessing {file}...")
            filepath = os.path.join(in_dir, file)
            base_name = os.path.splitext(file)[0]
            savepath = os.path.join(out_dir, base_name + ".csv")

            df = backend.transcribe_file(filepath, smokescreen=smokescreen, args=args)
            df.to_csv(savepath, index=False)

            backend.save_raw_output(out_dir, base_name)


# ─── Backward-compatible wrappers ───

def run_whisper(in_dir, out_dir, args=dict(),
                use_gpu=False, smokescreen=False, force_recompute=False, verbose=False):
    run_transcription(in_dir, out_dir, 'whisper', args=args,
                      use_gpu=use_gpu, smokescreen=smokescreen,
                      force_recompute=force_recompute, verbose=verbose)


def run_whisperx(in_dir, out_dir, args=dict(),
                 use_gpu=False, smokescreen=False, force_recompute=False, verbose=False):
    run_transcription(in_dir, out_dir, 'whisperx', args=args,
                      use_gpu=use_gpu, smokescreen=smokescreen,
                      force_recompute=force_recompute, verbose=verbose)


def run_assemblyai(in_dir, out_dir, args=dict(),
                   use_gpu=False, smokescreen=False, force_recompute=False, verbose=False):
    run_transcription(in_dir, out_dir, 'assemblyai', args=args,
                      use_gpu=use_gpu, smokescreen=smokescreen,
                      force_recompute=force_recompute, verbose=verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Automated Speech Annotation using AI Models')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the input directory.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory.')
    parser.add_argument('--use_gpu', action='store_true', default=False, help='Flag to use GPU.')
    parser.add_argument('--backend', type=str, default='whisperx',
                        choices=['whisper', 'whisperx', 'assemblyai'],
                        help='Transcription backend to use (default: whisperx)')
    parser.add_argument('--whisper_only', action='store_true', default=False,
                        help='DEPRECATED: Use --backend whisper instead.')
    parser.add_argument('--smokescreen', action='store_true', default=False,
                        help='Flag for smokescreen runs for fast testing.')
    parser.add_argument('--force_recompute', action='store_true', default=False,
                        help='Whether to force result recomputation.')
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"The directory {args.input_dir} does not exist.")
        sys.exit(1)

    if not os.path.isdir(args.input_dir):
        print(f"The path {args.input_dir} is not a directory.")
        sys.exit(1)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created the output directory: {args.output_dir}")

    backend = args.backend
    if args.whisper_only:
        warnings.warn("--whisper_only is deprecated. Use --backend whisper instead.", DeprecationWarning)
        backend = 'whisper'

    run_transcription(args.input_dir, args.output_dir, backend,
                      use_gpu=args.use_gpu, smokescreen=args.smokescreen,
                      force_recompute=args.force_recompute, verbose=True)
