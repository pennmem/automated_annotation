#!/bin/bash
# Remove auto-generated .ann and .csv files from session directories that were
# auto-annotated. A session is considered auto-annotated if it contains an
# AUTOMATED_ANNOT marker file or a model output subdirectory (whisperx_out,
# whisper_out, assemblyai_out) with .csv files.
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

# Check if a directory is an auto-annotated session
is_auto_annotated() {
    local dir="$1"
    # Check for AUTOMATED_ANNOT marker
    [ -f "$dir/AUTOMATED_ANNOT" ] && return 0
    # Check for model output csv files
    for out_dir in whisperx_out whisper_out assemblyai_out; do
        if [ -d "$dir/$out_dir" ] && ls "$dir/$out_dir"/*.csv &>/dev/null; then
            return 0
        fi
    done
    return 1
}

process_session() {
    local dir="$1"
    if ! is_auto_annotated "$dir"; then
        return
    fi

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

    # Remove the marker file if present
    if [ -f "$dir/AUTOMATED_ANNOT" ]; then
        echo "  Removing marker: $dir/AUTOMATED_ANNOT"
        rm "$dir/AUTOMATED_ANNOT"
    fi
}

# Find session directories and process them
if [ -n "$SUBJECT" ] && [ -z "$EXPERIMENT" ]; then
    # Subject only: search all experiments for that subject
    find "$SEARCH_PATH" -type d -path "*/$SUBJECT/session_*" | while read -r dir; do
        process_session "$dir"
    done
else
    find "$SEARCH_PATH" -type d -name "session_*" | while read -r dir; do
        process_session "$dir"
    done
fi
