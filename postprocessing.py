import glob
import os
import re
import wave
import pandas as pd
from abc import ABC, abstractmethod


class OutputRule(ABC):
    """Rule applied to transcription output DataFrame after transcribe_file()."""

    @abstractmethod
    def apply(self, df, context):
        """Apply rule to a transcription DataFrame.

        Args:
            df: DataFrame with columns Word, Onset, Offset, Probability
            context: dict with session metadata (wordpool, experiment, subject, etc.)

        Returns:
            Modified DataFrame.
        """
        pass


def _load_word2vec(model_name='glove-wiki-gigaword-50'):
    """Lazy-load a gensim KeyedVectors model (cached after first call)."""
    if not hasattr(_load_word2vec, '_model'):
        import gensim.downloader as api
        _load_word2vec._model = api.load(model_name)
    return _load_word2vec._model


def _semantic_similarity(word_a, word_b, threshold=0.5):
    """Return True if word_a and word_b are semantically similar via word2vec."""
    model = _load_word2vec()
    a = word_a.lower().replace('_', ' ').split()
    b = word_b.lower().replace('_', ' ').split()
    # Use only tokens present in the model vocabulary
    a = [t for t in a if t in model]
    b = [t for t in b if t in model]
    if not a or not b:
        return False
    try:
        return model.n_similarity(a, b) >= threshold
    except (KeyError, ZeroDivisionError):
        return False


class UpperCase(OutputRule):
    """Normalize words to uppercase."""

    def apply(self, df, context):
        df = df.copy()
        df['Word'] = df['Word'].str.strip().str.upper()
        return df


class SuffixStripping(OutputRule):
    """Strip common suffixes (plurals, -ed) to match wordpool base forms."""

    def apply(self, df, context):
        wordpool = context.get('wordpool')
        if wordpool is None:
            return df
        wp_set = set(w.upper() for w in wordpool)
        df = df.copy()
        df['Word'] = df['Word'].apply(lambda w: self._strip(str(w).strip().upper(), wp_set))
        return df

    @staticmethod
    def _strip(upper, wp_set):
        if upper in wp_set:
            return upper
        # Strip suffix from ASR word to match wordpool (KEYS -> KEY)
        for suffix, strip_len in [('S', 1), ('ES', 2), ('ED', 2), ('TE', 2)]:
            if upper.endswith(suffix) and upper[:-strip_len] in wp_set:
                return upper[:-strip_len]
        # Add suffix to ASR word to match wordpool (SCISSOR -> SCISSORS)
        for suffix in ['S', 'ES']:
            if (upper + suffix) in wp_set:
                return upper + suffix
        return upper


class SemanticMatch(OutputRule):
    """Match words to wordpool entries via substring containment + word2vec similarity."""

    def apply(self, df, context):
        wordpool = context.get('wordpool')
        if wordpool is None:
            return df
        wp_set = set(w.upper() for w in wordpool)
        df = df.copy()
        df['Word'] = df['Word'].apply(lambda w: self._match(w.upper(), wp_set))
        return df

    @staticmethod
    def _match(upper, wp_set):
        if upper in wp_set:
            return upper
        best = None
        for wp_word in wp_set:
            if wp_word in upper and len(wp_word) >= 3:
                if best is None or len(wp_word) > len(best):
                    best = wp_word
        if best is not None and _semantic_similarity(upper, best):
            return best
        return upper



class ListWordPreference(OutputRule):
    """Boost probability for words that appear in the .lst files (presented items).

    Per annotation rules: when choosing between wordpool words, prefer
    the bolded set (words on the list the participant is recalling).
    """

    BOOST = 0.05

    def apply(self, df, context):
        list_words = context.get('list_words')
        if list_words is None:
            return df
        df = df.copy()
        df['Probability'] = df.apply(
            lambda row: min(1.0, row['Probability'] + self.BOOST)
            if row['Word'].upper() in list_words
            else row['Probability'],
            axis=1,
        )
        return df


class MultiWordMerge(OutputRule):
    """Merge consecutive ASR words that together form a multi-word pool item.

    E.g. if the wordpool contains TRACING_PAPER and the transcript has
    TRACING followed immediately by PAPER, they become a single TRACING_PAPER
    row using the onset of the first word, offset of the last, and the minimum
    probability across the merged words.

    Must run before WordpoolFilter so the merged token is present for lookup.
    """

    def apply(self, df, context):
        wordpool = context.get('wordpool')
        # print(any(w == 'DRYER_MACHINE' for w in wordpool))
        if not wordpool:
            return df
        multiword = [(w.replace('_', ' ').split(), w) for w in wordpool if '_' in w or ' ' in w]
        # print(any(w == 'DRYER_MACHINE' for _, w in multiword))

        if not multiword:
            return df

        rows = df.reset_index(drop=True).to_dict('records')
        merged = []
        i = 0
        while i < len(rows):
            matched = False
            for parts, full_word in sorted(multiword, key=lambda x: -len(x[0])):
                # print(parts)
                # print(full_word)
                n = len(parts)
                if i + n <= len(rows):
                    window = [str(rows[i + j]['Word']).strip().upper() for j in range(n)]
                    # print(window)
                    if window == parts:
                        print(window)
                        combined = rows[i].copy()
                        combined['Word'] = full_word
                        combined['Offset'] = rows[i + n - 1].get('Offset', combined.get('Offset'))
                        probs = [rows[i + j].get('Probability') for j in range(n)]
                        probs = [p for p in probs if p is not None and not pd.isna(p)]
                        combined['Probability'] = min(probs) if probs else float('nan')
                        merged.append(combined)
                        i += n
                        matched = True
                        break
            if not matched:
                merged.append(rows[i])
                i += 1

        return pd.DataFrame(merged) if merged else df.iloc[0:0].copy()


class WordpoolFilter(OutputRule):
    """Classify words relative to the wordpool.

    Per annotation rules:
    - Words in wordpool -> correct recall (keep as-is)
    - Very low confidence words -> vocalization ('<>')

    Low-confidence threshold can be tuned; defaults to 0.1.
    """

    VOCALIZATION_THRESHOLD = 0.1

    def apply(self, df, context):
        wordpool = context.get('wordpool')
        if wordpool is None:
            return df
        df = df.copy()
        df['Word'] = df.apply(
            lambda row: '<>'
            if (row['Probability'] < self.VOCALIZATION_THRESHOLD)
            else str(row['Word']).strip().upper(),
            axis=1,
        )
        return df


class LongDurationVocalization(OutputRule):
    """Insert vocalization marks for words lasting longer than 1 second.

    Per annotation rules: if a word lasts audibly longer than 1 second,
    score the beginning as usual, then put a vocalization mark '<>' at
    every full second after the onset.
    """

    MAX_DURATION_MS = 1000

    def apply(self, df, context):
        new_rows = []
        for _, row in df.iterrows():
            new_rows.append(row.to_dict())
            onset = row['Onset']
            offset = row['Offset']
            if pd.notna(onset) and pd.notna(offset):
                duration = offset - onset
                if duration > self.MAX_DURATION_MS:
                    # Add <> marks at each full second after onset
                    t = onset + self.MAX_DURATION_MS
                    while t <= offset:
                        ext = {
                            'Word': '<>',
                            'Onset': int(t),
                            'Offset': int(min(t + self.MAX_DURATION_MS, offset)),
                            'Probability': row['Probability'],
                        }
                        if 'Type' in df.columns:
                            ext['Type'] = 'Extension'
                        new_rows.append(ext)
                        t += self.MAX_DURATION_MS
        return pd.DataFrame(new_rows)


class OnsetAdjust(OutputRule):
    """Shift onsets back by 5ms to align with annotation conventions.

    Per annotation rules: the annotation mark should go 5ms BEFORE the
    onset of the vocalization.
    """

    SHIFT_MS = 5

    def apply(self, df, context):
        df = df.copy()
        if 'Onset' in df.columns:
            df['Onset'] = df['Onset'].apply(
                lambda t: max(0, t - self.SHIFT_MS) if pd.notna(t) else t
            )
        return df

class WordpoolIndex(OutputRule):
    """Add a wordpool index column mapping each word to its position in the wordpool."""

    def apply(self, df, context):
        wordpool = context.get('wordpool')
        if not wordpool:
            return df
        df = df.copy()
        w_indices = {}
        for i, w in enumerate(wordpool):
            key = w.upper()
            if key not in w_indices:
                w_indices[key] = i + 1
        df['item_num'] = df['Word'].str.upper().map(w_indices).fillna(-1).astype(int)
        return df



class EmptyVocalization(OutputRule):
    """Insert a vocalization mark for WAV files with no detected words.

    If the transcription produced an empty DataFrame (no vocalizations detected),
    create a single '<>' mark with onset at the last second of the WAV file and
    offset 500ms later. This ensures every WAV file produces at least one annotation.

    Requires context['wav_path'] to be set to the current WAV file path.
    """

    def apply(self, df, context):
        if not df.empty:
            return df
        wav_path = context.get('wav_path')
        if not wav_path or not os.path.exists(wav_path):
            return df
        try:
            with wave.open(wav_path, 'rb') as wf:
                duration_ms = int(wf.getnframes() / wf.getframerate() * 1000)
        except Exception:
            return df
        onset = max(0, duration_ms - 1000)
        offset = onset + 500
        return pd.DataFrame([{
            'Word': '<>',
            'Onset': onset,
            'Offset': offset,
            'Probability': 0.0,
        }])


# ─── Vocalization classification constants ──────────────────────────────────

FILLER_WORDS = {
    'UM', 'UH', 'AH', 'OH', 'OKAY', 'OK', 'YEAH', 'YES', 'NO', 'HM',
    'HMM', 'HMMM', 'MMM', 'MMMHH', 'RIGHT', 'WELL', 'SO', 'LIKE',
    'HELLO', 'HI', 'HEY', 'SORRY', 'NOW',
    'OM', 'AHEM', 'PFFT', 'BYE', 'THANKS', 'THANK', 'HUH', 'GOD',
    'DAMN', 'SHIT',
}

NOISE_WORDS = {
    'COUGH', 'INHALE', 'EXHALE', 'BEEP', 'SNEEZE', 'SIGH', 'GASP',
    'SNIFF', 'HICCUP', 'CLICK', 'BUZZ', 'STATIC', 'NOISE',
    'BREATHING', 'BREATH', 'GRUNT', 'GROAN', 'SNORE', 'YAWN',
}

SENTENCE_WORDS = {
    'I', 'IM', 'A', 'AN', 'THE', 'IS', 'IT', 'ITS', 'IN', 'ON', 'OF', 'TO',
    'DO', 'DONT', 'DID', 'DIDNT', 'WAS', 'WASNT', 'WERE', 'WERENT',
    'CAN', 'CANT', 'COULD', 'COULDNT', 'WILL', 'WONT', 'WOULD', 'WOULDNT',
    'SHOULD', 'SHOULDNT', 'HAVE', 'HAS', 'HAD', 'HAVENT', 'HASNT',
    'NOT', 'BUT', 'AND', 'OR', 'IF', 'THEN', 'THAT', 'THATS', 'THIS',
    'WHAT', 'WHERE', 'WHEN', 'WHY', 'HOW', 'WHO', 'WHICH',
    'FOR', 'WITH', 'FROM', 'AT', 'BY', 'ABOUT', 'INTO',
    'JUST', 'REALLY', 'VERY', 'SOME', 'ANY', 'ALL', 'EVERY',
    'MY', 'YOUR', 'YOURE', 'HIS', 'HER', 'OUR', 'THEIR',
    'ME', 'YOU', 'HE', 'SHE', 'WE', 'THEY', 'ONE',
    'KNOW', 'THINK', 'FEEL', 'REMEMBER', 'WANT', 'NEED', 'TRY',
    'TRYING', 'DOING', 'GOING', 'GETTING', 'SURE', 'ACTUALLY',
    'KEEP', 'TRACK', 'TURN', 'OFF', 'NORMAL', 'PEOPLE', 'USUALLY',
    'HARD', 'GOOD', 'GREAT', 'FINE', 'BACK', 'THERE',
    'MATH', 'LUNCH', 'SORRY', 'ABOUT', 'COME',
    'STAY', 'BREAK', 'MAYBE', 'GO', 'SAID', 'WATCHING', 'MEET',
    'ALREADY', 'LAST', 'UP', 'LOVE', 'SWEET', 'SHORT',
    'RATHER', 'MOMMY', 'MR', 'NAN',
}

EXPERIMENTER_PHRASES = [
    ['RECALL', 'ALL', 'THE', 'WORDS'],
    ['WORDS', 'YOU', 'CAN', 'REMEMBER'],
    ['YOURE', 'DOING', 'GREAT'],
    ['YOURE', 'GETTING'],
    ['LOT', 'OF', 'GOOD', 'NOTES'],
    ['COME', 'IN', 'ITS', 'OKAY'],
    ['COME', 'IN', 'ITS'],
    ['COME', 'IN'],
    ['COME', 'BACK'],
    ['ITS', 'OKAY', 'COME'],
    ['KEEP', 'TRYING', 'TO', 'REMEMBER'],
    ['JUST', 'KEEP', 'TRYING'],
    ['THATS', 'FINE', 'JUST', 'KEEP'],
    ['WANTED', 'TO', 'CHECK'],
    ['WANTED', 'FOR', 'LUNCH'],
    ['CAN', 'TURN', 'IT', 'OFF'],
]

PARTICIPANT_PHRASES = [
    ['I', 'DONT'], ['I', 'JUST'], ['I', 'CANT'], ['IM', 'SURE'],
    ['IS', 'THIS', 'NORMAL'],
]


# ─── Shared helpers ─────────────────────────────────────────────────────────

def _contains_phrases(words, phrases):
    """Return True if any phrase from *phrases* appears in *words*."""
    for phrase in phrases:
        plen = len(phrase)
        for start in range(len(words) - plen + 1):
            if words[start:start + plen] == phrase:
                return True
    return False


def _find_phrase_ranges(words, phrases):
    """Return list of (start, end) index ranges where phrases match in *words*."""
    ranges = []
    for phrase in phrases:
        plen = len(phrase)
        for start in range(len(words) - plen + 1):
            if words[start:start + plen] == phrase:
                ranges.append((start, start + plen))
    return ranges


def _is_sentence_word(word):
    """True if *word* is a common English word, not a guess attempt."""
    w = word.upper()
    return w in SENTENCE_WORDS or w in FILLER_WORDS


def _word_upper(row):
    """Safely extract uppercase word from a row."""
    return str(row['Word']).strip().upper()


def _group_non_wordpool_runs(df, wp_set, filler_set,
                             sentence_gap_ms=1000, filler_gap_ms=300):
    """Yield (start, end) index tuples of consecutive non-wordpool row runs."""
    n = len(df)
    i = 0
    while i < n:
        w = _word_upper(df.iloc[i])
        if w in wp_set:
            i += 1
            continue
        run_start = i
        j = i
        while j < n:
            wj = _word_upper(df.iloc[j])
            if wj in wp_set:
                break
            if j > run_start:
                gap = df.iloc[j]['Onset'] - df.iloc[j - 1]['Offset']
                if gap > sentence_gap_ms:
                    break
                if wj in filler_set and gap > filler_gap_ms:
                    break
            j += 1
        yield (run_start, j)
        i = j


# ─── Vocalization sub-rules ─────────────────────────────────────────────────

class BreathingNoiseDetection(OutputRule):
    """Mark ASR-transcribed breathing and noise artifacts as vocalizations.

    Any word matching NOISE_WORDS (COUGH, INHALE, EXHALE, BEEP, etc.)
    becomes ``<>`` with ``Type='Noise'``.
    """

    def apply(self, df, context):
        wp_set = set(w.upper() for w in (context.get('wordpool') or []))
        noise_set = NOISE_WORDS - wp_set
        df = df.copy()
        if 'Type' not in df.columns:
            df['Type'] = ''
        for i in range(len(df)):
            t = str(df.at[df.index[i], 'Type']).strip()
            if t and t != 'nan':
                continue
            w = _word_upper(df.iloc[i])
            if w in noise_set:
                df.at[df.index[i], 'Word'] = '<>'
                df.at[df.index[i], 'Type'] = 'Noise'
        return df


class FillerDetection(OutputRule):
    """Mark standalone filler words (um, uh, okay, …) as vocalizations.

    Filler words not in the wordpool become ``<>`` with ``Type='Filler'``.
    """

    def apply(self, df, context):
        wp_set = set(w.upper() for w in (context.get('wordpool') or []))
        filler_set = FILLER_WORDS - wp_set
        df = df.copy()
        if 'Type' not in df.columns:
            df['Type'] = ''
        for i in range(len(df)):
            t = str(df.at[df.index[i], 'Type']).strip()
            if t and t != 'nan':
                continue
            w = _word_upper(df.iloc[i])
            if w in filler_set:
                df.at[df.index[i], 'Word'] = '<>'
                df.at[df.index[i], 'Type'] = 'Filler'
        return df


class ExperimenterSpeechFilter(OutputRule):
    """Detect and mark experimenter speech for removal.

    Multi-word runs are scanned for EXPERIMENTER_PHRASES. Matched spans
    (plus surrounding non-participant words) are marked ``Type='Experimenter'``.
    The orchestrating VocalizationClassifier drops these rows.

    Also handles manual ``experimenter_overrides`` from context.
    """

    SENTENCE_GAP_MS = 1000

    def apply(self, df, context):
        wp_set = set(w.upper() for w in (context.get('wordpool') or []))
        filler_set = FILLER_WORDS - wp_set
        df = df.copy()
        if 'Type' not in df.columns:
            df['Type'] = ''

        # Manual overrides
        exp_overrides = context.get('experimenter_overrides', {})
        list_num = context.get('list_num')
        override_rows = exp_overrides.get(list_num, set()) if list_num is not None else set()
        for i in range(len(df)):
            if i in override_rows:
                df.at[df.index[i], 'Type'] = 'Experimenter'

        # Phrase-based detection on non-wordpool runs
        for run_s, run_e in _group_non_wordpool_runs(df, wp_set, filler_set):
            words = [_word_upper(df.iloc[k]) for k in range(run_s, run_e)]
            if len(words) < 2:
                continue
            segments = self._split(df, run_s, run_e, words)
            for seg_type, seg_s, seg_e in segments:
                if seg_type == 'experimenter':
                    for k in range(seg_s, seg_e):
                        df.at[df.index[k], 'Type'] = 'Experimenter'
        return df

    def _split(self, df, start, end, words):
        upper = [w.upper() for w in words]
        if _contains_phrases(upper, EXPERIMENTER_PHRASES) and not _contains_phrases(upper, PARTICIPANT_PHRASES):
            return [('experimenter', start, end)]

        exp_ranges = _find_phrase_ranges(upper, EXPERIMENTER_PHRASES)
        if not exp_ranges:
            return [('participant', start, end)]

        exp_ranges.sort()
        merged = [list(exp_ranges[0])]
        for s, e in exp_ranges[1:]:
            if s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])

        for mi, (s, e) in enumerate(merged):
            while e < len(upper):
                actual = start + e
                if actual >= len(df):
                    break
                gap = df.iloc[actual]['Onset'] - df.iloc[actual - 1]['Offset']
                if gap > self.SENTENCE_GAP_MS:
                    break
                if _contains_phrases(upper[e:], PARTICIPANT_PHRASES):
                    break
                e += 1
            while s > 0 and upper[s - 1] in FILLER_WORDS:
                s -= 1
            merged[mi] = [s, e]

        merged.sort()
        final = [merged[0]]
        for s, e in merged[1:]:
            if s <= final[-1][1]:
                final[-1][1] = max(final[-1][1], e)
            else:
                final.append([s, e])

        segments = []
        pos = 0
        for exp_s, exp_e in final:
            if pos < exp_s:
                segments.append(('participant', start + pos, start + exp_s))
            segments.append(('experimenter', start + exp_s, start + exp_e))
            pos = exp_e
        if pos < len(upper):
            segments.append(('participant', start + pos, start + len(upper)))
        return segments


class SentenceDetection(OutputRule):
    """Group consecutive common English words into sentence vocalizations.

    Multi-word participant speech (non-wordpool, non-filler, non-noise) where
    each word is in SENTENCE_WORDS gets merged into a single ``<>`` event
    with ``Type='Sentence'``. Intrusion-looking words within a sentence are
    extracted and left for IntrusionClassification.
    """

    def apply(self, df, context):
        wp_set = set(w.upper() for w in (context.get('wordpool') or []))
        filler_set = FILLER_WORDS - wp_set
        df = df.copy()
        if 'Type' not in df.columns:
            df['Type'] = ''

        for run_s, run_e in _group_non_wordpool_runs(df, wp_set, filler_set):
            # Skip runs already fully typed (experimenter, noise, filler)
            untyped = [k for k in range(run_s, run_e)
                       if str(df.at[df.index[k], 'Type']).strip() in ('', 'nan')]
            if len(untyped) < 2:
                continue

            # Check if this is a multi-word sentence span
            sentence_parts = []
            for k in untyped:
                w = _word_upper(df.iloc[k])
                if _is_sentence_word(w):
                    sentence_parts.append(k)

            if len(sentence_parts) < 2:
                continue

            # Group consecutive sentence parts
            groups = []
            current = [sentence_parts[0]]
            for k in sentence_parts[1:]:
                if k == current[-1] + 1 or (k - current[-1] == 2 and
                        str(df.at[df.index[current[-1] + 1], 'Type']).strip() not in ('', 'nan')):
                    current.append(k)
                else:
                    groups.append(current)
                    current = [k]
            groups.append(current)

            for group in groups:
                if len(group) < 2:
                    # Single sentence word -> mark as Sentence
                    k = group[0]
                    w = _word_upper(df.iloc[k])
                    if w not in filler_set:
                        df.at[df.index[k], 'Word'] = '<>'
                        df.at[df.index[k], 'Type'] = 'Sentence'
                else:
                    # Merge: keep first row's onset, last row's offset
                    first, last = group[0], group[-1]
                    df.at[df.index[first], 'Word'] = '<>'
                    df.at[df.index[first], 'Offset'] = int(df.iloc[last]['Offset'])
                    df.at[df.index[first], 'Type'] = 'Sentence'
                    # Mark rest for removal
                    for k in group[1:]:
                        df.at[df.index[k], 'Type'] = '_merged'
        # Drop merged rows
        df = df[df['Type'] != '_merged'].reset_index(drop=True)
        return df


class IntrusionClassification(OutputRule):
    """Mark remaining untyped non-wordpool words as intrusions.

    Catch-all: any row still without a Type that is not in the wordpool
    gets ``Type='Intrusion'``. Common sentence words get ``Type='Sentence'``
    and ``Word='<>'`` instead (single-word sentence fragments).
    """

    def apply(self, df, context):
        wp_set = set(w.upper() for w in (context.get('wordpool') or []))
        filler_set = FILLER_WORDS - wp_set
        df = df.copy()
        if 'Type' not in df.columns:
            df['Type'] = ''
        for i in range(len(df)):
            t = str(df.at[df.index[i], 'Type']).strip()
            if t and t != 'nan':
                continue
            w = _word_upper(df.iloc[i])
            if w in wp_set:
                continue
            if _is_sentence_word(w):
                df.at[df.index[i], 'Word'] = '<>'
                df.at[df.index[i], 'Type'] = 'Filler' if w in filler_set else 'Sentence'
            else:
                df.at[df.index[i], 'Type'] = 'Intrusion'
        return df


class ProperNounDetection(OutputRule):
    """Reclassify intrusions that match known first names as proper nouns.

    Loads a name list from ``dependencies/first_names.txt`` and marks
    matching Intrusion rows as ``<>`` with ``Type='ProperNoun'``.
    Names that appear in the wordpool are skipped.
    """

    _DEFAULT_PATH = os.path.join(os.path.dirname(__file__), 'dependencies', 'first_names.txt')

    def __init__(self, names_path=None):
        path = names_path or self._DEFAULT_PATH
        self._names = set()
        if os.path.exists(path):
            with open(path) as f:
                self._names = {line.strip().upper() for line in f if line.strip()}

    def apply(self, df, context):
        if not self._names:
            return df
        wp_set = set(w.upper() for w in (context.get('wordpool') or []))
        names = self._names - wp_set
        df = df.copy()
        for i in range(len(df)):
            if str(df.at[df.index[i], 'Type']).strip() != 'Intrusion':
                continue
            w = _word_upper(df.iloc[i])
            if w in names:
                df.at[df.index[i], 'Word'] = '<>'
                df.at[df.index[i], 'Type'] = 'ProperNoun'
        return df


# ─── Orchestrator ────────────────────────────────────────────────────────────

class VocalizationClassifier(OutputRule):
    """Orchestrate vocalization sub-rules to classify each ASR word.

    Runs sub-rules in order:
    1. BreathingNoiseDetection — catch noise tokens first
    2. FillerDetection — catch filler words
    3. ExperimenterSpeechFilter — mark experimenter spans
    4. SentenceDetection — merge sentence runs
    5. IntrusionClassification — catch-all for remaining
    6. ProperNounDetection — reclassify name intrusions

    Then drops Experimenter rows and ensures clean output.

    Must run **after** UpperCase / SuffixStripping / SemanticMatch.
    """

    def __init__(self, disable=None):
        self._disable = set(disable or [])
        self._sub_rules = []
        rule_classes = [
            ('BreathingNoiseDetection', BreathingNoiseDetection),
            ('FillerDetection', FillerDetection),
            ('ExperimenterSpeechFilter', ExperimenterSpeechFilter),
            ('SentenceDetection', SentenceDetection),
            ('IntrusionClassification', IntrusionClassification),
            ('ProperNounDetection', ProperNounDetection),
        ]
        for name, cls in rule_classes:
            if name not in self._disable:
                self._sub_rules.append(cls())

    def apply(self, df, context):
        wordpool = context.get('wordpool')
        if wordpool is None:
            return df

        df = df.copy()
        if 'Type' not in df.columns:
            df['Type'] = ''

        n = len(df)
        if n == 0:
            return df

        # Run each sub-rule in sequence
        for rule in self._sub_rules:
            df = rule.apply(df, context)

        # Drop experimenter speech rows
        df = df[df['Type'] != 'Experimenter'].reset_index(drop=True)

        # Ensure int types for timing columns
        for col in ('Onset', 'Offset'):
            if col in df.columns:
                df[col] = df[col].astype(int)
        return df


OUTPUT_RULE_REGISTRY = {
    'UpperCase': UpperCase,
    'SuffixStripping': SuffixStripping,
    'SemanticMatch': SemanticMatch,
    'MultiWordMerge': MultiWordMerge,
    'ListWordPreference': ListWordPreference,
    'WordpoolFilter': WordpoolFilter,
    'VocalizationClassifier': VocalizationClassifier,
    'BreathingNoiseDetection': BreathingNoiseDetection,
    'FillerDetection': FillerDetection,
    'ExperimenterSpeechFilter': ExperimenterSpeechFilter,
    'SentenceDetection': SentenceDetection,
    'IntrusionClassification': IntrusionClassification,
    'ProperNounDetection': ProperNounDetection,
    'OnsetAdjust': OnsetAdjust,
    'LongDurationVocalization': LongDurationVocalization,
    'WordpoolIndex': WordpoolIndex,
    'EmptyVocalization': EmptyVocalization,
}


def build_output_rules(args):
    """Instantiate output rules from args config.

    Args:
        args: dict that may contain "rules" key with list of rule configs.
              Each config is a dict with at least "name" key.
              e.g. [{"name": "wordpool_filter"}]

    Returns:
        List of OutputRule instances.
    """
    rule_configs = args.get('rules', []) if args else []
    rules = []
    for cfg in rule_configs:
        name = cfg if isinstance(cfg, str) else cfg['name']
        if name not in OUTPUT_RULE_REGISTRY:
            raise ValueError(
                f"Unknown output rule '{name}'. Available: {list(OUTPUT_RULE_REGISTRY.keys())}"
            )
        rules.append(OUTPUT_RULE_REGISTRY[name]())

    # EmptyVocalization always runs as a check on all sessions
    if not any(isinstance(r, EmptyVocalization) for r in rules):
        rules.append(EmptyVocalization())

    return rules


def apply_output_rules(df, rules, context):
    """Apply a list of output rules to a transcription DataFrame."""
    for rule in rules:
        if df.empty and not isinstance(rule, EmptyVocalization):
            continue
        df = rule.apply(df, context)
    return df


def build_context(in_dir, args=None):
    """Build session context from the input directory path.

    Parses experiment and subject from the path structure:
        /data/eeg/scalp/ltp/{experiment}/{subject}/session_{N}/

    Loads the wordpool if a wordpool_pattern is provided in args["rules_config"].
    Falls back to auto-discovering *wordpool*.txt in the subject directory.
    Also loads .lst files from the session directory (presented items).

    Returns:
        dict with keys: experiment, subject, session, wordpool (set or None),
        list_words (set or None)
    """
    context = {
        'experiment': None,
        'subject': None,
        'session': None,
        'wordpool': None,
    }

    # Parse path components
    match = re.search(
        r'/data/eeg/scalp/ltp/([^/]+)/([^/]+)/session_(\d+)',
        in_dir
    )
    if match:
        context['experiment'] = match.group(1)
        context['subject'] = match.group(2)
        context['session'] = int(match.group(3))

    # Load wordpool
    wordpool_path = None

    # Check args for explicit wordpool path template
    rules_config = (args or {}).get('rules_config', {})
    if 'wordpool_path' in rules_config and context['experiment'] and context['subject']:
        wordpool_path = rules_config['wordpool_path'].format(
            experiment=context['experiment'],
            subject=context['subject'],
        )

    # Auto-discover: look for any *wordpool*.txt in the subject directory
    if wordpool_path is None and context['experiment'] and context['subject']:
        subject_dir = f"/data/eeg/scalp/ltp/{context['experiment']}/{context['subject']}"
        matches = glob.glob(os.path.join(subject_dir, '*wordpool*.txt'))
        if matches:
            wordpool_path = matches[0]

    if wordpool_path and os.path.exists(wordpool_path):
        with open(wordpool_path) as f:
            context['wordpool'] = [line.strip().upper() for line in f if line.strip()]

    # Load .lst files from session directory (presented items)
    list_words = set()
    if context['experiment'] and context['subject'] and context['session'] is not None:
        session_dir = f"/data/eeg/scalp/ltp/{context['experiment']}/{context['subject']}/session_{context['session']}"
        for lst_file in glob.glob(os.path.join(session_dir, '*.lst')):
            with open(lst_file) as f:
                for line in f:
                    word = line.strip().upper()
                    if word:
                        list_words.add(word)
    context['list_words'] = list_words if list_words else None

    # Merge list words into wordpool so they always pass filtering
    if list_words:
        if context['wordpool'] is None:
            context['wordpool'] = []
        existing = set(context['wordpool'])
        context['wordpool'].extend(w for w in list_words if w not in existing)

    return context
