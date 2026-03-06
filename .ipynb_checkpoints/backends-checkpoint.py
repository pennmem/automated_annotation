import os
import string
import json
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


STANDARD_COLUMNS = ['Word', 'Onset', 'Offset', 'Probability']


class TranscriptionBackend(ABC):
    """Abstract base class for transcription backends."""

    output_subdir: str = None

    @abstractmethod
    def load_model(self, use_gpu, smokescreen, args):
        """Load model/resources. Called once before processing files."""
        pass

    @abstractmethod
    def transcribe_file(self, filepath, smokescreen, args):
        """Transcribe a single audio file.

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

    def load_model(self, use_gpu, smokescreen, args):
        import whisperx

        device = "cuda:0" if use_gpu else "cpu"
        compute_type = "float16" if use_gpu else "int8"
        model_size = "large-v2" if not smokescreen else 'tiny.en'
        self.model = whisperx.load_model(model_size, device, compute_type=compute_type)
        self.device = device

    def transcribe_file(self, filepath, smokescreen, args):
        import whisperx

        audio = whisperx.load_audio(filepath)
        if smokescreen:
            audio = audio[:len(audio) // 3]

        transcribe_args = args.get('transcribe', {})
        result = self.model.transcribe(audio, batch_size=self.batch_size, **transcribe_args)

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

    def load_model(self, use_gpu, smokescreen, args):
        import whisperx

        device = "cuda:0" if use_gpu else "cpu"
        self.device = device

        whisper_dir = None
        if args and 'whisperx_args' in args and 'whisper_dir' in args['whisperx_args']:
            whisper_dir = args['whisperx_args']['whisper_dir']
        self.whisper_dir = whisper_dir

        if not whisper_dir:
            compute_type = "float16" if use_gpu else "int8"
            model_size = "large-v2" if not smokescreen else 'tiny.en'
            self.model = whisperx.load_model(model_size, device, compute_type=compute_type)

        self.model_a, self.metadata = whisperx.load_align_model(
            language_code="en", device=device
        )

    def transcribe_file(self, filepath, smokescreen, args):
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
            result = self.model.transcribe(audio, batch_size=self.batch_size, **transcribe_args)

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

    def load_model(self, use_gpu, smokescreen, args):
        import assemblyai as aai

        api_key = os.environ.get('ASSEMBLYAI_API_KEY')
        if not api_key:
            raise EnvironmentError(
                "ASSEMBLYAI_API_KEY environment variable is not set. "
                "Please set it before using the AssemblyAI backend."
            )
        aai.settings.api_key = api_key

        config = aai.TranscriptionConfig(
            punctuate=False,
            format_text=False,
            language_code="en",
        )
        self.transcriber = aai.Transcriber(config=config)

    def transcribe_file(self, filepath, smokescreen, args):
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
