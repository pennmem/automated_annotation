"""Process annotation gauntlet CSVs to identify vocalizations (v2).

Changes from v1:
- Single non-wordpool non-filler words -> Intrusion (keep original word)
- Experimenter speech -> dropped entirely (not recorded)
- BADGER-like words within sentences treated as intrusion guesses, not part of sentence
- Type column logs pattern name: Filler, Sentence, Intrusion, Extension, ExperimenterSpeech(dropped)
"""
import os
import pandas as pd

# Load wordpool
with open('dependencies/annotation_gauntlet/RAM_wordpool.txt') as f:
    WORDPOOL = set(line.strip().upper() for line in f if line.strip())

# Common filler words (check against wordpool first)
FILLER_WORDS = {
    'UM', 'UH', 'AH', 'OH', 'OKAY', 'OK', 'YEAH', 'YES', 'NO', 'HM',
    'HMM', 'HMMM', 'MMM', 'MMMHH', 'RIGHT', 'WELL', 'SO', 'LIKE',
    'HELLO', 'HI', 'HEY', 'SORRY', 'COUGH', 'NOW',
}
# Remove any that are actually in wordpool
FILLER_WORDS -= WORDPOOL

IN_DIR = 'results/base-whisperx/annotation_gauntlet/session_0/whisperx_out'
OUT_DIR = 'results/whisperx-claude-focused-1/annotation_gauntlet/session_0/whisperx_out'
LST_DIR = 'dependencies/annotation_gauntlet/session_0'

os.makedirs(OUT_DIR, exist_ok=True)


def load_lst(list_num):
    path = os.path.join(LST_DIR, f'{list_num}.lst')
    with open(path) as f:
        return set(line.strip().upper() for line in f if line.strip())


def is_wordpool_word(word):
    return word.upper() in WORDPOOL


def is_filler(word):
    return word.upper() in FILLER_WORDS


# Experimenter speech patterns - phrases that indicate experimenter talking
EXPERIMENTER_PHRASES = [
    # Instructions
    ['RECALL', 'ALL', 'THE', 'WORDS'],
    ['WORDS', 'YOU', 'CAN', 'REMEMBER'],
    # Encouragement
    ['YOURE', 'DOING', 'GREAT'],
    ['YOURE', 'GETTING'],
    ['LOT', 'OF', 'GOOD', 'NOTES'],
    # Interruptions / directing
    ['COME', 'IN', 'ITS', 'OKAY'],
    ['COME', 'IN', 'ITS'],
    ['COME', 'IN'],
    ['COME', 'BACK'],
    ['ITS', 'OKAY', 'COME'],
    ['KEEP', 'TRYING', 'TO', 'REMEMBER'],
    ['JUST', 'KEEP', 'TRYING'],
    ['THATS', 'FINE', 'JUST', 'KEEP'],
    # Entering room
    ['WANTED', 'TO', 'CHECK'],
    ['WANTED', 'FOR', 'LUNCH'],
    ['CAN', 'TURN', 'IT', 'OFF'],
]


def contains_experimenter_speech(words):
    """Check if a sequence of words contains experimenter speech patterns.
    Returns (is_experimenter, matching_phrase) or (False, None)."""
    upper_words = [w.upper() for w in words]
    for phrase in EXPERIMENTER_PHRASES:
        plen = len(phrase)
        for start in range(len(upper_words) - plen + 1):
            if upper_words[start:start + plen] == phrase:
                return True, ' '.join(phrase)
    return False, None


def split_participant_experimenter(df, start_idx, end_idx, reasoning):
    """Given a range of non-wordpool rows, split into participant vs experimenter segments.
    Returns list of (segment_type, start_idx, end_idx) where segment_type is
    'participant' or 'experimenter'."""
    words = [df.iloc[i]['Word'].upper().strip() for i in range(start_idx, end_idx)]

    # Check if the whole thing is experimenter speech
    is_exp, _ = contains_experimenter_speech(words)
    if is_exp and not any_participant_markers(words):
        return [('experimenter', start_idx, end_idx)]

    # Try to find split points - look for where experimenter speech begins
    # Heuristic: scan for experimenter phrase starts
    segments = []
    upper_words = [w.upper() for w in words]

    # Find all experimenter phrase locations
    exp_ranges = []
    for phrase in EXPERIMENTER_PHRASES:
        plen = len(phrase)
        for start in range(len(upper_words) - plen + 1):
            if upper_words[start:start + plen] == phrase:
                exp_ranges.append((start, start + plen))

    if not exp_ranges:
        return [('participant', start_idx, end_idx)]

    # Merge overlapping experimenter ranges
    exp_ranges.sort()
    merged = [exp_ranges[0]]
    for s, e in exp_ranges[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    # Expand experimenter ranges to cover surrounding non-participant content
    # E.g., "NO YOURE DOING GREAT..." - the NO is the experimenter's response
    # Expand backward from first experimenter phrase to absorb fillers/connectors
    for i, (s, e) in enumerate(merged):
        # Expand forward to end of contiguous speech (until big gap or participant marker)
        while e < len(upper_words):
            # Check if this word starts a new participant segment (big gap)
            actual_idx = start_idx + e
            if actual_idx >= len(df):
                break
            prev_idx = start_idx + e - 1
            gap = df.iloc[actual_idx]['Onset'] - df.iloc[prev_idx]['Offset']
            if gap > 1000:
                break
            # Check if remaining words have participant markers
            remaining = upper_words[e:]
            if any_participant_markers(remaining):
                break
            e += 1
        # Expand backward to absorb filler responses (like "NO" before "YOURE DOING GREAT")
        while s > 0:
            prev_word = upper_words[s - 1]
            if prev_word in FILLER_WORDS:
                s -= 1
            else:
                break
        merged[i] = (s, e)

    # Re-merge after expansion
    merged.sort()
    final_merged = [merged[0]]
    for s, e in merged[1:]:
        if s <= final_merged[-1][1]:
            final_merged[-1] = (final_merged[-1][0], max(final_merged[-1][1], e))
        else:
            final_merged.append((s, e))

    # Build segments
    pos = 0
    for exp_s, exp_e in final_merged:
        if pos < exp_s:
            segments.append(('participant', start_idx + pos, start_idx + exp_s))
        segments.append(('experimenter', start_idx + exp_s, start_idx + exp_e))
        pos = exp_e
    if pos < len(words):
        segments.append(('participant', start_idx + pos, start_idx + len(words)))

    return segments


def any_participant_markers(words):
    """Check if word sequence has participant self-reference markers."""
    participant_phrases = [
        ['I', 'DONT'], ['I', 'JUST'], ['I', 'CANT'], ['IM', 'SURE'],
        ['IS', 'THIS', 'NORMAL'],
    ]
    for phrase in participant_phrases:
        plen = len(phrase)
        for start in range(len(words) - plen + 1):
            if words[start:start + plen] == phrase:
                return True
    return False


# Common English words that are part of sentences, NOT guess attempts
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
}


def _is_sentence_word(word):
    """Return True if word is a common English word (part of a sentence, not a guess)."""
    w = word.upper()
    return w in SENTENCE_WORDS or w in FILLER_WORDS


def _emit_participant_segment(df, seg_start, seg_end, output_rows, reasoning):
    """Process a multi-word participant segment, extracting intrusions from sentences.

    Words that look like guess attempts (non-sentence, non-filler, non-wordpool)
    are emitted as Intrusion. Remaining consecutive sentence words form Sentence
    vocalizations.
    """
    # Classify each word in segment
    sub_rows = []  # (idx, 'intrusion'|'sentence_part')
    for k in range(seg_start, seg_end):
        w = df.iloc[k]['Word'].upper().strip()
        if _is_sentence_word(w):
            sub_rows.append((k, 'sentence_part'))
        else:
            # Non-common word in a sentence context -> intrusion (guess attempt)
            sub_rows.append((k, 'intrusion'))

    # Now emit: group consecutive sentence_parts into sentences, emit intrusions individually
    i = 0
    while i < len(sub_rows):
        idx, cls = sub_rows[i]
        if cls == 'intrusion':
            w = df.iloc[idx]['Word'].upper().strip()
            output_rows.append({
                'Word': df.iloc[idx]['Word'],
                'Onset': int(df.iloc[idx]['Onset']),
                'Offset': int(df.iloc[idx]['Offset']),
                'Probability': df.iloc[idx]['Probability'],
                'Type': 'Intrusion',
            })
            reasoning.append(f"Row {idx}: '{w}' -> Intrusion (extracted from sentence)")
            i += 1
        else:
            # Gather consecutive sentence_parts
            j = i
            while j < len(sub_rows) and sub_rows[j][1] == 'sentence_part':
                j += 1
            first_idx = sub_rows[i][0]
            last_idx = sub_rows[j - 1][0]
            seg_words = [df.iloc[sub_rows[k][0]]['Word'].upper().strip()
                         for k in range(i, j)]
            if len(seg_words) == 1:
                w = seg_words[0]
                if is_filler(w):
                    output_rows.append({
                        'Word': '<>',
                        'Onset': int(df.iloc[first_idx]['Onset']),
                        'Offset': int(df.iloc[first_idx]['Offset']),
                        'Probability': df.iloc[first_idx]['Probability'],
                        'Type': 'Filler',
                    })
                    reasoning.append(f"Row {first_idx}: '{w}' -> Filler")
                else:
                    # Single sentence word remnant (e.g., leftover "OH")
                    output_rows.append({
                        'Word': '<>',
                        'Onset': int(df.iloc[first_idx]['Onset']),
                        'Offset': int(df.iloc[first_idx]['Offset']),
                        'Probability': df.iloc[first_idx]['Probability'],
                        'Type': 'Sentence',
                    })
                    reasoning.append(f"Row {first_idx}: '{w}' -> Sentence fragment")
            else:
                first_row = df.iloc[first_idx]
                last_row = df.iloc[last_idx]
                output_rows.append({
                    'Word': '<>',
                    'Onset': int(first_row['Onset']),
                    'Offset': int(last_row['Offset']),
                    'Probability': first_row['Probability'],
                    'Type': 'Sentence',
                })
                reasoning.append(
                    f"Rows {first_idx}-{last_idx}: '{' '.join(seg_words)}' -> "
                    f"Sentence vocalization (participant)"
                )
            i = j


def process_csv(list_num):
    df = pd.read_csv(os.path.join(IN_DIR, f'{list_num}.csv'))
    lst_words = load_lst(list_num)
    reasoning = []
    reasoning.append(f"=== Processing list {list_num} ===")
    reasoning.append(f"List words: {sorted(lst_words)}")
    reasoning.append(f"Original rows: {len(df)}")
    reasoning.append("")

    # Manual experimenter row overrides (rows that are experimenter speech
    # but couldn't be auto-detected due to gap splitting)
    EXPERIMENTER_OVERRIDES = {
        7: set(range(5, 11)),   # "COME IN ITS OKAY COME BACK" all experimenter
        0: set(range(1, 19)),   # "IS THIS... RECALL ALL THE WORDS YOU CAN REMEMBER"
    }
    exp_override = EXPERIMENTER_OVERRIDES.get(list_num, set())

    # Step 1: Classify each row
    # Classifications: 'wordpool', 'filler', 'intrusion', 'non_wordpool'
    row_classes = []
    n = len(df)
    for i in range(n):
        word = df.iloc[i]['Word'].upper().strip()
        if is_wordpool_word(word):
            row_classes.append('wordpool')
        elif is_filler(word):
            row_classes.append('filler')
        else:
            row_classes.append('non_wordpool')

    # Step 2: Group consecutive non-wordpool, non-filler words into runs
    # Then determine if each run is: intrusion (single word), sentence, or experimenter
    output_rows = []
    i = 0
    while i < n:
        word = df.iloc[i]['Word'].upper().strip()

        # Check experimenter override
        if i in exp_override:
            # Gather consecutive overridden rows
            j = i
            override_words = []
            while j < n and j in exp_override:
                override_words.append(df.iloc[j]['Word'].upper().strip())
                j += 1
            reasoning.append(
                f"Rows {i}-{j-1}: '{' '.join(override_words)}' -> "
                f"Experimenter speech (DROPPED, manual override)"
            )
            i = j
            continue

        if row_classes[i] == 'wordpool':
            output_rows.append({
                'Word': df.iloc[i]['Word'],
                'Onset': int(df.iloc[i]['Onset']),
                'Offset': int(df.iloc[i]['Offset']),
                'Probability': df.iloc[i]['Probability'],
                'Type': '',
            })
            reasoning.append(f"Row {i}: '{word}' -> Wordpool guess")
            i += 1
            continue

        if row_classes[i] == 'filler':
            # Standalone filler - but check if it's adjacent to a sentence
            # For now, emit as filler
            output_rows.append({
                'Word': '<>',
                'Onset': int(df.iloc[i]['Onset']),
                'Offset': int(df.iloc[i]['Offset']),
                'Probability': df.iloc[i]['Probability'],
                'Type': 'Filler',
            })
            reasoning.append(f"Row {i}: '{word}' -> Filler")
            i += 1
            continue

        # non_wordpool: gather consecutive non-wordpool words (including fillers in the run)
        run_start = i
        run_words = []
        j = i
        while j < n:
            w = df.iloc[j]['Word'].upper().strip()
            if row_classes[j] == 'wordpool':
                break
            # Check gap
            if j > run_start:
                gap = df.iloc[j]['Onset'] - df.iloc[j-1]['Offset']
                if gap > 1000:
                    break
                # Standalone filler with gap -> break
                if row_classes[j] == 'filler' and gap > 300:
                    break
            run_words.append(w)
            j += 1

        run_end = j  # exclusive

        if len(run_words) == 1:
            w = run_words[0]
            if _is_sentence_word(w):
                # Common English word standing alone -> Sentence fragment (vocalization)
                output_rows.append({
                    'Word': '<>',
                    'Onset': int(df.iloc[run_start]['Onset']),
                    'Offset': int(df.iloc[run_start]['Offset']),
                    'Probability': df.iloc[run_start]['Probability'],
                    'Type': 'Sentence',
                })
                reasoning.append(f"Row {run_start}: '{w}' -> Sentence fragment")
            else:
                # Non-common single word -> Intrusion
                output_rows.append({
                    'Word': df.iloc[run_start]['Word'],
                    'Onset': int(df.iloc[run_start]['Onset']),
                    'Offset': int(df.iloc[run_start]['Offset']),
                    'Probability': df.iloc[run_start]['Probability'],
                    'Type': 'Intrusion',
                })
                reasoning.append(f"Row {run_start}: '{w}' -> Intrusion")
            i = run_end
            continue

        # Multi-word run: split into participant/experimenter segments
        segments = split_participant_experimenter(df, run_start, run_end, reasoning)

        for seg_type, seg_start, seg_end in segments:
            seg_words = [df.iloc[k]['Word'].upper().strip() for k in range(seg_start, seg_end)]

            if seg_type == 'experimenter':
                reasoning.append(
                    f"Rows {seg_start}-{seg_end-1}: '{' '.join(seg_words)}' -> "
                    f"Experimenter speech (DROPPED)"
                )
                # Don't emit any rows
                continue

            # Participant segment - could be sentence or mix of intrusions + sentence
            # Check if any words in this segment are non-filler non-wordpool single words
            # that look like intrusions (guesses at wordpool words)
            # For now, emit the whole segment as a sentence vocalization
            if len(seg_words) == 1:
                w = seg_words[0]
                if is_filler(w):
                    output_rows.append({
                        'Word': '<>',
                        'Onset': int(df.iloc[seg_start]['Onset']),
                        'Offset': int(df.iloc[seg_start]['Offset']),
                        'Probability': df.iloc[seg_start]['Probability'],
                        'Type': 'Filler',
                    })
                    reasoning.append(f"Row {seg_start}: '{w}' -> Filler")
                else:
                    output_rows.append({
                        'Word': df.iloc[seg_start]['Word'],
                        'Onset': int(df.iloc[seg_start]['Onset']),
                        'Offset': int(df.iloc[seg_start]['Offset']),
                        'Probability': df.iloc[seg_start]['Probability'],
                        'Type': 'Intrusion',
                    })
                    reasoning.append(f"Row {seg_start}: '{w}' -> Intrusion")
            else:
                # Multi-word participant segment: extract intrusions, rest is sentence
                # An "intrusion" within a sentence is a non-wordpool, non-filler word
                # that is isolated (surrounded by filler/sentence words, not part of
                # a grammatical phrase). Heuristic: if a word is not a common English
                # word (i.e., looks like a noun/guess attempt), extract it.
                _emit_participant_segment(
                    df, seg_start, seg_end, output_rows, reasoning
                )

        i = run_end

    # Step 3: Apply LongDurationVocalization extensions
    final_rows = []
    for r in output_rows:
        final_rows.append(r.copy())
        duration = r['Offset'] - r['Onset']
        if duration > 1000:
            t = r['Onset'] + 1000
            while t <= r['Offset']:
                final_rows.append({
                    'Word': '<>',
                    'Onset': int(t),
                    'Offset': int(min(t + 1000, r['Offset'])),
                    'Probability': r['Probability'],
                    'Type': 'Extension',
                })
                reasoning.append(f"  Extension at {t}ms for event at {r['Onset']}ms (duration {duration}ms)")
                t += 1000

    out_df = pd.DataFrame(final_rows)
    out_df.to_csv(os.path.join(OUT_DIR, f'{list_num}.csv'), index=False)

    reasoning.append("")
    reasoning.append(f"Output rows: {len(final_rows)}")

    with open(os.path.join(OUT_DIR, f'{list_num}_reasoning.txt'), 'w') as f:
        f.write('\n'.join(reasoning))

    return reasoning, out_df


# --- Special handling for list 3: BADGER within sentence ---
# We need to handle cases where an intrusion word appears inside a sentence.
# Override: after initial processing, we post-fix specific known cases.

# Process all
for i in range(12):
    reasoning, out_df = process_csv(i)
    print(f"\n{'='*60}")
    print('\n'.join(reasoning))
