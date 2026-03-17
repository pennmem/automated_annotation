#!/bin/bash
# Remove auto-generated .ann files from session directories that contain
# an AUTOMATED_ANNOT marker file.
#
# Usage: ./remove_auto_ann.sh [-e EXPERIMENT] [-s SUBJECT] [-n SESSION]
#   -e EXPERIMENT   Only process this experiment (e.g. ltpFR3)
#   -s SUBJECT      Only process this subject (e.g. LTP123)
#   -n SESSION      Only process this session number (requires -e and -s)
#
# Examples:
#   ./remove_auto_ann.sh                         # all sessions
#   ./remove_auto_ann.sh -e ltpFR3               # all subjects in ltpFR3
#   ./remove_auto_ann.sh -s LTP123               # LTP123 across all experiments
#   ./remove_auto_ann.sh -e ltpFR3 -s LTP123     # LTP123 in ltpFR3 only
#   ./remove_auto_ann.sh -e ltpFR3 -s LTP123 -n 0  # specific session

ROOT="/data/eeg/scalp/ltp"
EXPERIMENT=""
SUBJECT=""
SESSION=""

while getopts "e:s:n:" opt; do
    case "$opt" in
        e) EXPERIMENT="$OPTARG" ;;
        s) SUBJECT="$OPTARG" ;;
        n) SESSION="$OPTARG" ;;
        *)
            echo "Usage: $0 [-e EXPERIMENT] [-s SUBJECT] [-n SESSION]"
            exit 1
            ;;
    esac
done

# Session requires both experiment and subject
if [ -n "$SESSION" ] && { [ -z "$EXPERIMENT" ] || [ -z "$SUBJECT" ]; }; then
    echo "Error: -n SESSION requires both -e EXPERIMENT and -s SUBJECT"
    exit 1
fi

# Build the search path
SEARCH_PATH="$ROOT"
if [ -n "$EXPERIMENT" ] && [ -n "$SUBJECT" ] && [ -n "$SESSION" ]; then
    SEARCH_PATH="$ROOT/$EXPERIMENT/$SUBJECT/session_$SESSION"
elif [ -n "$EXPERIMENT" ] && [ -n "$SUBJECT" ]; then
    SEARCH_PATH="$ROOT/$EXPERIMENT/$SUBJECT"
elif [ -n "$EXPERIMENT" ]; then
    SEARCH_PATH="$ROOT/$EXPERIMENT"
fi

if [ ! -d "$SEARCH_PATH" ]; then
    echo "Error: directory not found: $SEARCH_PATH"
    exit 1
fi

# If only subject specified (no experiment), search all experiments for that subject
if [ -n "$SUBJECT" ] && [ -z "$EXPERIMENT" ]; then
    FIND_CMD=(find "$SEARCH_PATH" -path "*/$SUBJECT/*/AUTOMATED_ANNOT")
else
    FIND_CMD=(find "$SEARCH_PATH" -name AUTOMATED_ANNOT)
fi

"${FIND_CMD[@]}" | while read -r marker; do
    dir="$(dirname "$marker")"

    echo "Processing: $dir"

    # Remove .ann files
    ann_files=("$dir"/*.ann)
    if [ -e "${ann_files[0]}" ]; then
        echo "  Removing .ann files"
        rm "${ann_files[@]}"
    fi

    # Remove .csv files from model output subdirectories
    for out_dir in whisperx_out whisper_out assemblyai_out; do
        if [ -d "$dir/$out_dir" ]; then
            csv_files=("$dir/$out_dir"/*.csv)
            if [ -e "${csv_files[0]}" ]; then
                echo "  Removing .csv files in: $dir/$out_dir"
                rm "${csv_files[@]}"
            fi
        fi
    done

    # Remove the marker file itself
    echo "  Removing marker: $marker"
    rm "$marker"
done
