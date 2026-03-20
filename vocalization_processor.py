"""Process annotation gauntlet CSVs to identify vocalizations."""
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
OUT_DIR = 'results/whisperx-claude-focused/annotation_gauntlet/session_0/whisperx_out'
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

def process_csv(list_num):
    df = pd.read_csv(os.path.join(IN_DIR, f'{list_num}.csv'))
    lst_words = load_lst(list_num)
    reasoning = []
    reasoning.append(f"=== Processing list {list_num} ===")
    reasoning.append(f"List words: {sorted(lst_words)}")
    reasoning.append(f"Original rows: {len(df)}")
    reasoning.append("")

    # Classify each row
    classifications = []  # (index, 'keep'|'filler'|'sentence_start'|'sentence_mid')
    n = len(df)
    i = 0
    while i < n:
        word = df.iloc[i]['Word'].upper().strip()

        # Check if it's a wordpool word (potential guess)
        if is_wordpool_word(word):
            # It's a wordpool word - check some edge cases
            # KEYS -> KEY (suffix), ICES -> ICE, GLASSES -> GLASS, BADGER -> BADGE?
            # For now keep as guess
            classifications.append((i, 'keep', None))
            i += 1
            continue

        # Check if filler
        if is_filler(word):
            classifications.append((i, 'filler', None))
            reasoning.append(f"Row {i}: '{word}' -> Filler vocalization")
            i += 1
            continue

        # Not wordpool and not filler - could be part of a sentence
        # Look ahead to find consecutive non-wordpool words
        sentence_start = i
        sentence_words = [word]
        j = i + 1
        while j < n:
            next_word = df.iloc[j]['Word'].upper().strip()
            # If next word is wordpool, stop
            if is_wordpool_word(next_word):
                break
            # If gap > 1000ms between words, likely separate utterances
            gap = df.iloc[j]['Onset'] - df.iloc[j-1]['Offset']
            if gap > 1000:
                break
            # If it's a standalone filler with a gap, treat separately
            if is_filler(next_word) and gap > 300:
                break
            sentence_words.append(next_word)
            j += 1

        if len(sentence_words) == 1:
            # Single non-wordpool, non-filler word
            # Could be a filler variant or a sentence of 1
            if is_filler(word):
                classifications.append((i, 'filler', None))
                reasoning.append(f"Row {i}: '{word}' -> Filler")
            else:
                # Single non-wordpool word - treat as filler if it looks like one
                classifications.append((i, 'filler', None))
                reasoning.append(f"Row {i}: '{word}' -> Single non-wordpool word, marked as Filler")
            i += 1
        else:
            # Multiple consecutive non-wordpool words = sentence
            classifications.append((sentence_start, 'sentence_start', sentence_words))
            reasoning.append(f"Rows {sentence_start}-{j-1}: '{' '.join(sentence_words)}' -> Sentence vocalization")
            for k in range(sentence_start + 1, j):
                classifications.append((k, 'sentence_mid', None))
            i = j

    # Build output dataframe
    output_rows = []
    for idx, cls, extra in classifications:
        row = df.iloc[idx]
        if cls == 'keep':
            output_rows.append({
                'Word': row['Word'],
                'Onset': int(row['Onset']),
                'Offset': int(row['Offset']),
                'Probability': row['Probability'],
                'Type': '',
            })
        elif cls == 'filler':
            output_rows.append({
                'Word': '<>',
                'Onset': int(row['Onset']),
                'Offset': int(row['Offset']),
                'Probability': row['Probability'],
                'Type': 'Filler',
            })
        elif cls == 'sentence_start':
            # Find last word in sentence
            last_idx = idx + len(extra) - 1
            last_row = df.iloc[last_idx]
            output_rows.append({
                'Word': '<>',
                'Onset': int(row['Onset']),
                'Offset': int(last_row['Offset']),
                'Probability': row['Probability'],
                'Type': 'Sentence',
            })
        # sentence_mid rows are absorbed into the sentence_start

    # Now apply LongDurationVocalization: if any event > 1000ms, add extension
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

    # Patterns
    reasoning.append("")
    reasoning.append(f"Output rows: {len(final_rows)}")

    with open(os.path.join(OUT_DIR, f'{list_num}_reasoning.txt'), 'w') as f:
        f.write('\n'.join(reasoning))

    return reasoning, out_df

# Process all
all_patterns = []
for i in range(12):
    reasoning, out_df = process_csv(i)
    print(f"\n{'='*60}")
    print('\n'.join(reasoning))

# Summary of patterns
print("\n\n=== PATTERNS OBSERVED ===")
print("""
1. SUFFIX VARIANTS: Words like KEYS (KEY+S), ICES (ICE+S), GLASSES (GLASS+ES),
   BADGER (BADGE+R), MATTRESS, BLUE are not in wordpool but close to wordpool words.
   These may be misrecognitions or the participant saying a variant.

2. REPEATED GUESSES: Some words are repeated multiple times (e.g., ROPE in list 6,
   BOY/BLOOM/BLUE repeated in list 2, CHAIN repeated in list 3). The participant
   may be rehearsing or second-guessing.

3. CONVERSATIONAL INTERRUPTIONS: Lists 1, 2, 5, 7, 8 contain what appear to be
   conversations with the experimenter mid-recall (e.g., "I just wanted to check
   what you wanted for lunch", "sorry I can turn it off", "come in it's okay").

4. SELF-COMMENTARY: Phrases like "I don't know", "I don't remember any",
   "is this normal with people usually" are the participant talking about the task
   rather than recalling words.

5. MATH REFERENCE: In list 5, "really hard math" suggests the participant is
   commenting on the distractor task between lists.

6. EXPERIMENTER SPEECH: Some speech may be from the experimenter rather than the
   participant (e.g., "you're doing great", "come in it's okay", "recall all the
   words you can remember").
""")
