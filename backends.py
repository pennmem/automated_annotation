import os
import string
import json
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


STANDARD_COLUMNS = ['Word', 'Onset', 'Offset', 'Probability']

# Max tokens for Whisper's initial_prompt conditioning window is ~224 tokens.
# We budget roughly 200 words to stay safe.
_MAX_PROMPT_WORDS = 200


def _build_initial_prompt(context):
    """Build an initial_prompt string from context for Whisper conditioning.

    Puts file-specific list words first (highest priority), then session-wide
    list words, then remaining wordpool words. Truncates to fit Whisper's
    prompt window.
    """
    if context is None:
        return None

    words = []
    seen = set()

    # 1. File-specific list words (the exact words presented in this trial)
    for w in sorted(context.get('file_list_words', None) or []):
        if w not in seen:
            words.append(w)
            seen.add(w)

    # 2. Session-wide list words (all presented words across trials)
    for w in sorted(context.get('list_words', None) or []):
        if w not in seen:
            words.append(w)
            seen.add(w)

    # 3. Remaining wordpool words
    for w in sorted(context.get('wordpool', None) or []):
        if w not in seen:
            words.append(w)
            seen.add(w)

    if not words:
        return None

    # Truncate and join as comma-separated list
    words = words[:_MAX_PROMPT_WORDS]
    return ", ".join(words)


def _parse_device(device_str):
    """Parse a device string like 'cuda:2' into ('cuda', 2) for ctranslate2/whisperx.

    Returns (device, device_index) tuple. For 'cpu' or 'cuda', device_index is 0.
    """
    if ':' in device_str:
        base, idx = device_str.split(':', 1)
        return base, int(idx)
    return device_str, 0


class TranscriptionBackend(ABC):
    """Abstract base class for transcription backends."""

    output_subdir: str = None

    @abstractmethod
    def load_model(self, smokescreen, args, **kwargs):
        """Load model/resources. Called once before processing files."""
        pass

    @abstractmethod
    def transcribe_file(self, filepath, smokescreen, args, context=None):
        """Transcribe a single audio file.

        Args:
            filepath: Path to the .wav file.
            smokescreen: If True, truncate audio for fast testing.
            args: Dict with backend-specific arguments.
            context: Optional dict from build_context() with wordpool,
                     list_words, and file_list_words for biasing transcription.

        Returns:
            pd.DataFrame with columns: Word, Onset, Offset, Probability
        """
        pass

    def save_raw_output(self, out_dir, base_name):
        """Optional hook to save raw output (e.g. JSON). Default: no-op."""
        pass


class WhisperBackend(TranscriptionBackend):
    """Whisper (via whisperx) -- transcription only, no word-level alignment."""

    output_subdir = 'whisper_out'

    def __init__(self):
        self.model = None
        self.batch_size = 16

    def load_model(self, smokescreen, args, use_gpu=False, device=None, model_size=None, **kwargs):
        import torch
        import whisperx

        if device is None:
            if use_gpu and torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        dev, dev_idx = _parse_device(device)
        print(f"WhisperBackend: device={dev}, device_index={dev_idx}")
        if model_size is None:
            model_size = "large-v2" if not smokescreen else 'tiny.en'
        try:
            compute_type = "float16" if dev == "cuda" else "int8"
            self.model = whisperx.load_model(model_size, dev, device_index=dev_idx, compute_type=compute_type, language="en")
        except ValueError:
            print(f"WARNING: Failed to load model on {dev}:{dev_idx}, falling back to CPU")
            dev = "cpu"
            dev_idx = 0
            self.model = whisperx.load_model(model_size, dev, compute_type="int8", language="en")
        self.device = dev

    def transcribe_file(self, filepath, smokescreen, args, context=None):
        import whisperx

        audio = whisperx.load_audio(filepath)
        if smokescreen:
            audio = audio[:len(audio) // 3]

        transcribe_args = args.get('transcribe', {})
        prompt = _build_initial_prompt(context)
        if prompt and 'initial_prompt' not in transcribe_args:
            transcribe_args['initial_prompt'] = prompt
        result = self.model.transcribe(audio, batch_size=self.batch_size, language="en", **transcribe_args)

        self._last_raw_result = result

        words = []
        for segment in result['segments']:
            text = segment["text"].strip()
            for w in text.split(" "):
                w = w.translate(str.maketrans('', '', string.punctuation)).upper()
                if w.strip():
                    words.append(w)

        return pd.DataFrame({
            'Word': words,
            'Onset': [np.nan] * len(words),
            'Offset': [np.nan] * len(words),
            'Probability': [np.nan] * len(words),
        })

    def save_raw_output(self, out_dir, base_name):
        if hasattr(self, '_last_raw_result'):
            with open(os.path.join(out_dir, base_name + '.json'), 'w') as f:
                json.dump(self._last_raw_result, f)


class WhisperXBackend(TranscriptionBackend):
    """WhisperX -- transcription + wav2vec2 forced alignment for word-level timing."""

    output_subdir = 'whisperx_out'

    def __init__(self):
        self.model = None
        self.model_a = None
        self.metadata = None
        self.batch_size = 16
        self.whisper_dir = None

    def load_model(self, smokescreen, args, use_gpu=False, device=None, model_size=None, **kwargs):
        import torch
        import whisperx

        if device is None:
            if use_gpu and torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        dev, dev_idx = _parse_device(device)
        if dev == "cuda":
            # Pin this process to a single physical GPU via CUDA_VISIBLE_DEVICES.
            # After this, "cuda:0" / "cuda" refers to the selected physical GPU.
            os.environ["CUDA_VISIBLE_DEVICES"] = str(dev_idx)
            torch.cuda.set_device(0)
        self.device = "cuda" if dev == "cuda" else "cpu"
        print(f"WhisperXBackend: physical GPU={dev_idx}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

        whisper_dir = None
        if args and 'whisperx_args' in args and 'whisper_dir' in args['whisperx_args']:
            whisper_dir = args['whisperx_args']['whisper_dir']
        self.whisper_dir = whisper_dir

        if not whisper_dir:
            if model_size is None:
                model_size = "large-v2" if not smokescreen else 'tiny.en'
            try:
                compute_type = "float16" if dev == "cuda" else "int8"
                self.model = whisperx.load_model(model_size, "cuda", device_index=0, compute_type=compute_type, language="en")
            except ValueError:
                print(f"WARNING: Failed to load model on GPU {dev_idx}, falling back to CPU")
                self.device = "cpu"
                self.model = whisperx.load_model(model_size, "cpu", compute_type="int8", language="en")

        self.model_a, self.metadata = whisperx.load_align_model(
            language_code="en", device=self.device
        )

    def transcribe_file(self, filepath, smokescreen, args, context=None):
        import whisperx

        audio = whisperx.load_audio(filepath)
        if smokescreen:
            audio = audio[:len(audio) // 3]

        base_name = os.path.splitext(os.path.basename(filepath))[0]

        if self.whisper_dir:
            with open(os.path.join(self.whisper_dir, base_name + '.json'), 'r') as f:
                result = json.load(f)
        else:
            transcribe_args = args.get('transcribe', {})
            prompt = _build_initial_prompt(context)
            if prompt:
                self.model.options.initial_prompt = prompt
            result = self.model.transcribe(audio, batch_size=self.batch_size, language="en", **transcribe_args)

        result = whisperx.align(
            result["segments"], self.model_a, self.metadata,
            audio, self.device, return_char_alignments=False
        )

        words, onsets, offsets, confidence = [], [], [], []
        for segment in result["segments"]:
            for element in segment['words']:
                word = element['word'].translate(
                    str.maketrans('', '', string.punctuation)
                ).upper()
                words.append(word)
                onsets.append(int(element['start'] * 1000))
                offsets.append(int(element['end'] * 1000))
                confidence.append(element['score'])

        return pd.DataFrame({
            'Word': words,
            'Onset': onsets,
            'Offset': offsets,
            'Probability': confidence,
        })


class AssemblyAIBackend(TranscriptionBackend):
    """AssemblyAI cloud API -- transcription with word-level timestamps."""

    output_subdir = 'assemblyai_out'

    def __init__(self):
        self.transcriber = None

    def load_model(self, smokescreen, args, speech_models=None, **kwargs):
        import assemblyai as aai

        api_key = os.environ.get('ASSEMBLY_AI_KEY')
        if not api_key:
            raise EnvironmentError(
                "ASSEMBLYAI_API_KEY environment variable is not set. "
                "Please set it before using the AssemblyAI backend."
            )
        aai.settings.api_key = api_key

        if speech_models is None:
            speech_models = ["universal-2"] if smokescreen else ["universal-2", "universal-3-pro"]
        elif not isinstance(speech_models, list):
            speech_models = [speech_models]

        config = aai.TranscriptionConfig(
            speech_models=speech_models,
            punctuate=False,
            format_text=False,
            language_code="en",
        )
        self.transcriber = aai.Transcriber(config=config)

    def transcribe_file(self, filepath, smokescreen, args, context=None):
        if context is not None:
            import assemblyai as aai
            boost_words = []
            # File-specific list words get highest priority
            for w in sorted(context.get('file_list_words', None) or []):
                boost_words.append(w.capitalize())
            # Then session-wide list words
            for w in sorted(context.get('list_words', None) or []):
                cap = w.capitalize()
                if cap not in boost_words:
                    boost_words.append(cap)
            # Then remaining wordpool words
            for w in sorted(context.get('wordpool', None) or []):
                cap = w.capitalize()
                if cap not in boost_words:
                    boost_words.append(cap)
            if boost_words:
                config = aai.TranscriptionConfig(
                    speech_models=self.transcriber.config.speech_models,
                    punctuate=False,
                    format_text=False,
                    language_code="en",
                    word_boost=boost_words,
                    boost_param="high",
                )
                self.transcriber.config = config

        transcript = self.transcriber.transcribe(filepath)

        if transcript.status == "error":
            raise RuntimeError(
                f"AssemblyAI transcription failed for {filepath}: {transcript.error}"
            )

        words, onsets, offsets, confidence = [], [], [], []
        if transcript.words:
            for w in transcript.words:
                word = w.text.translate(
                    str.maketrans('', '', string.punctuation)
                ).upper()
                if word.strip():
                    words.append(word)
                    onsets.append(int(w.start))
                    offsets.append(int(w.end))
                    confidence.append(float(w.confidence))

        return pd.DataFrame({
            'Word': words,
            'Onset': onsets,
            'Offset': offsets,
            'Probability': confidence,
        })


BACKEND_REGISTRY = {
    'whisper': WhisperBackend,
    'whisperx': WhisperXBackend,
    'assemblyai': AssemblyAIBackend,
}


def get_backend(name):
    """Instantiate a backend by name. Matches if the key is contained in the name."""
    name_lower = name.lower()
    # Check exact match first, then substring
    if name_lower in BACKEND_REGISTRY:
        return BACKEND_REGISTRY[name_lower]()
    for key, cls in BACKEND_REGISTRY.items():
        if key in name_lower:
            return cls()
    raise ValueError(
        f"Unknown backend '{name}'. Available: {list(BACKEND_REGISTRY.keys())}"
    )
