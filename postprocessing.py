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


class WordpoolFilter(OutputRule):
    """Replace transcribed words not in the subject's wordpool with '<>'."""

    def apply(self, df, context):
        wordpool = context.get('wordpool')
        if wordpool is None:
            return df
        df = df.copy()
        df['Word'] = df['Word'].apply(
            lambda w: w if w.upper() in wordpool else '<>'
        )
        return df


OUTPUT_RULE_REGISTRY = {
    'wordpool_filter': WordpoolFilter,
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
    Falls back to auto-discovering wasnorm_wordpool.txt in the subject directory.

    Returns:
        dict with keys: experiment, subject, session, wordpool (set or None)
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

    # Auto-discover: look for wasnorm_wordpool.txt in the subject directory
    if wordpool_path is None and context['experiment'] and context['subject']:
        auto_path = f"/data/eeg/scalp/ltp/{context['experiment']}/{context['subject']}/wasnorm_wordpool.txt"
        if os.path.exists(auto_path):
            wordpool_path = auto_path

    if wordpool_path and os.path.exists(wordpool_path):
        with open(wordpool_path) as f:
            context['wordpool'] = {line.strip().upper() for line in f if line.strip()}

    return context
