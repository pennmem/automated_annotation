#!/bin/bash
# Remove auto-generated .ann and .csv files from session directories that were
# auto-annotated. A session is considered auto-annotated if it contains an
# AUTOMATED_ANNOT marker file or a model output subdirectory (whisperx_out,
# whisper_out, assemblyai_out) with .csv files.
#
# Usage: ./remove_auto_ann.sh [-e EXPERIMENT] [-s SUBJECT] [-n SESSION]
#                             [-c] [-a] [-p] [-P]
#   -e EXPERIMENT   Only process this experiment (e.g. ltpFR3)
#   -s SUBJECT      Only process this subject (e.g. LTP123)
#   -n SESSION      Only process this session number (requires -e and -s)
#   -c              Remove .csv files from model output subdirectories
#   -a              Remove .ann files
#   -p              Also delete .par files (instead of skipping sessions with them)
#   -P              Ignore .par files (don't skip sessions that have them, but don't delete them)
#
#   If neither -c nor -a is given, both are removed (default behavior).
#
# Examples:
#   ./remove_auto_ann.sh                         # all sessions, remove .ann + .csv
#   ./remove_auto_ann.sh -e ltpFR3               # all subjects in ltpFR3
#   ./remove_auto_ann.sh -s LTP123               # LTP123 across all experiments
#   ./remove_auto_ann.sh -e ltpFR3 -s LTP123     # LTP123 in ltpFR3 only
#   ./remove_auto_ann.sh -e ltpFR3 -s LTP123 -n 0  # specific session
#   ./remove_auto_ann.sh -c                      # only remove .csv files
#   ./remove_auto_ann.sh -a                      # only remove .ann files
#   ./remove_auto_ann.sh -p                      # also delete .par files
#   ./remove_auto_ann.sh -P                      # ignore .par files (don't skip, don't delete)

ROOT="/data/eeg/scalp/ltp"
EXPERIMENT=""
SUBJECT=""
SESSION=""
RM_CSV=0
RM_ANN=0
RM_PAR=0        # -p: delete .par files
IGNORE_PAR=0    # -P: ignore .par files (don't skip, don't delete)

while getopts "e:s:n:capP" opt; do
    case "$opt" in
        e) EXPERIMENT="$OPTARG" ;;
        s) SUBJECT="$OPTARG" ;;
        n) SESSION="$OPTARG" ;;
        c) RM_CSV=1 ;;
        a) RM_ANN=1 ;;
        p) RM_PAR=1 ;;
        P) IGNORE_PAR=1 ;;
        *)
            echo "Usage: $0 [-e EXPERIMENT] [-s SUBJECT] [-n SESSION] [-c] [-a] [-p] [-P]"
            exit 1
            ;;
    esac
done

# Default: if neither -c nor -a specified, remove both
if [ "$RM_CSV" -eq 0 ] && [ "$RM_ANN" -eq 0 ]; then
    RM_CSV=1
    RM_ANN=1
fi

# -p and -P are mutually exclusive
if [ "$RM_PAR" -eq 1 ] && [ "$IGNORE_PAR" -eq 1 ]; then
    echo "Error: -p and -P are mutually exclusive"
    exit 1
fi

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
    # Remanant of old system which uses model_out.csv as indicator fo automation
    # for out_dir in whisperx_out whisper_out assemblyai_out; do
    #     if [ -d "$dir/$out_dir" ] && ls "$dir/$out_dir"/*.csv &>/dev/null; then
    #         return 1
    #     fi
    # done
    return 1
}

process_session() {
    local dir="$1"
    if ! is_auto_annotated "$dir"; then
        return
    fi

    # Skip sessions that have been corrected via parsync
    if [ -f "$dir/CORRECTED_ANNOT" ]; then
        echo "Skipping (corrected annotations): $dir"
        return
    fi

    # Handle .par files
    if ls "$dir"/*.par &>/dev/null; then
        if [ "$RM_PAR" -eq 1 ]; then
            echo "  Deleting .par files in: $dir"
        elif [ "$IGNORE_PAR" -eq 0 ]; then
            echo "Skipping (has .par files): $dir"
            return
        fi
        # IGNORE_PAR=1: just continue without skipping or deleting
    fi

    echo "Processing: $dir"

    # Delete .par files if requested
    if [ "$RM_PAR" -eq 1 ]; then
        par_files=("$dir"/*.par)
        if [ -e "${par_files[0]}" ]; then
            echo "  Removing .par files"
            rm "${par_files[@]}"
        fi
    fi

    # Remove .ann files
    if [ "$RM_ANN" -eq 1 ]; then
        ann_files=("$dir"/*.ann)
        if [ -e "${ann_files[0]}" ]; then
            echo "  Removing .ann files"
            rm "${ann_files[@]}"
        fi
    fi

    # Remove .csv files from model output subdirectories and session directory
    if [ "$RM_CSV" -eq 1 ]; then
        for out_dir in whisperx_out whisper_out assemblyai_out; do
            if [ -d "$dir/$out_dir" ]; then
                csv_files=("$dir/$out_dir"/*.csv)
                if [ -e "${csv_files[0]}" ]; then
                    echo "  Removing .csv files in: $dir/$out_dir"
                    rm "${csv_files[@]}"
                fi
            fi
        done
        # Also remove auto_whisperx_*.csv files in the session directory itself
        auto_csv_files=("$dir"/auto_whisperx_*.csv)
        if [ -e "${auto_csv_files[0]}" ]; then
            echo "  Removing auto_whisperx_*.csv files in: $dir"
            rm "${auto_csv_files[@]}"
        fi
    fi

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
