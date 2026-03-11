import glob
import os
import re
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


class PluralNormalization(OutputRule):
    """Normalize plurals and common suffixed forms to their wordpool base.

    Per annotation rules: ignore plurals and score as the singular version.
    Also handles longer words containing the target (e.g. 'telephone' -> 'phone').
    """

    def apply(self, df, context):
        wordpool = context.get('wordpool')
        if wordpool is None:
            return df
        df = df.copy()
        df['Word'] = df['Word'].apply(lambda w: self._normalize(w, wordpool))
        return df

    @staticmethod
    def _normalize(word, wordpool):
        upper = word.upper()
        if upper in wordpool:
            return upper

        # Strip trailing 's' for plurals (e.g. DOGS -> DOG)
        if upper.endswith('S') and upper[:-1] in wordpool:
            return upper[:-1]

        # Strip 'es' suffix (e.g. FOXES -> FOX)
        if upper.endswith('ES') and upper[:-2] in wordpool:
            return upper[:-2]

        # Strip 'ed' suffix (e.g. PARKED -> PARK)
        if upper.endswith('ED') and upper[:-2] in wordpool:
            return upper[:-2]

        # Check if any wordpool word is contained in this word
        # (e.g. TELEPHONE contains PHONE). Pick the longest match.
        best = None
        for wp_word in wordpool:
            if wp_word in upper and len(wp_word) >= 3:
                if best is None or len(wp_word) > len(best):
                    best = wp_word
        if best is not None:
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


class WordpoolFilter(OutputRule):
    """Classify words relative to the wordpool.

    Per annotation rules:
    - Words in wordpool -> correct recall (keep as-is)
    - Words NOT in wordpool but identifiable -> intrusion (keep the word)
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
            lambda row: '<>' if row['Probability'] < self.VOCALIZATION_THRESHOLD
            else row['Word'],
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
                        new_rows.append({
                            'Word': '<>',
                            'Onset': int(t),
                            'Offset': int(min(t + self.MAX_DURATION_MS, offset)),
                            'Probability': row['Probability'],
                        })
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


OUTPUT_RULE_REGISTRY = {
    'plural_normalization': PluralNormalization,
    'list_word_preference': ListWordPreference,
    'wordpool_filter': WordpoolFilter,
    'long_duration_vocalization': LongDurationVocalization,
    'onset_adjust': OnsetAdjust,
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
    return rules


def apply_output_rules(df, rules, context):
    """Apply a list of output rules to a transcription DataFrame."""
    for rule in rules:
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
            context['wordpool'] = {line.strip().upper() for line in f if line.strip()}

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
            context['wordpool'] = set()
        context['wordpool'] |= list_words

    return context
