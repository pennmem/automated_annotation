#!/bin/bash
# Remove auto-generated .ann and .csv files from session directories that contain
# output files for a given model type.
# Also removes the corresponding .ann files from the parse_files SVN working copy and commits.
#
# Usage: ./remove_auto_ann.sh MODEL_TYPE [ROOT_DIR] [PARSE_FILES_DIR]
#   MODEL_TYPE      required: whisperx, whisper, or assemblyai
#   ROOT_DIR        defaults to /data/eeg/scalp/ltp
#   PARSE_FILES_DIR defaults to ~/parse_files

if [ -z "$1" ]; then
    echo "Usage: $0 MODEL_TYPE [ROOT_DIR] [PARSE_FILES_DIR]"
    echo "  MODEL_TYPE: whisperx, whisper, or assemblyai"
    exit 1
fi

MODEL_TYPE="$1"
ROOT="${2:-/data/eeg/scalp/ltp}"
PARSE_FILES_DIR="${3:-$HOME/parse_files}"

# Map model type to output directory pattern
case "$MODEL_TYPE" in
    whisperx)  OUT_DIR="whisperx_out" ;;
    whisper)   OUT_DIR="whisper_out" ;;
    assemblyai) OUT_DIR="assemblyai_out" ;;
    *)
        echo "Unknown model type: $MODEL_TYPE (expected whisperx, whisper, or assemblyai)"
        exit 1
        ;;
esac

find "$ROOT" -name AUTOMATED_ANNOT | while read -r marker; do
    dir="$(dirname "$marker")"

    # Only process if the marker matches the requested model type
    if ! grep -q "backend=$MODEL_TYPE" "$marker"; then
        continue
    fi

    echo "Processing: $dir (marker matches $MODEL_TYPE)"

    # Remove .ann files
    ann_files=("$dir"/*.ann)
    if [ -e "${ann_files[0]}" ]; then
        echo "  Removing .ann files"
        rm "${ann_files[@]}"

        # Mirror removal into parse_files SVN working copy
        # rel="$(realpath --relative-to="$ROOT" "$dir")"
        # pf_dir="$PARSE_FILES_DIR/$rel"
        # if [ -d "$pf_dir" ]; then
        #     pf_ann_files=("$pf_dir"/*.ann)
        #     if [ -e "${pf_ann_files[0]}" ]; then
        #         echo "  svn delete .ann files in: $pf_dir"
        #         svn delete "${pf_ann_files[@]}"
        #     fi
        # fi
    fi

    # Remove .csv files from the output subdirectory
    if [ -d "$dir/$OUT_DIR" ]; then
        out_csv_files=("$dir/$OUT_DIR"/*.csv)
        if [ -e "${out_csv_files[0]}" ]; then
            echo "  Removing .csv files in: $dir/$OUT_DIR"
            rm "${out_csv_files[@]}"
        fi
    fi

    # Remove the marker file itself
    echo "  Removing marker: $marker"
    rm "$marker"
done

# # Commit all deletions in one pass
# if svn status "$PARSE_FILES_DIR" | grep -q '^D'; then
#     echo "Committing SVN deletions in $PARSE_FILES_DIR ..."
#     svn commit "$PARSE_FILES_DIR" -m "Remove auto-generated .ann files ($MODEL_TYPE present)"
#     echo "SVN commit done."
# else
#     echo "No SVN deletions to commit."
# fi
