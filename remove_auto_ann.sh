#!/bin/bash
# Remove all .ann files from session directories that contain whisperx* files.
# Usage: ./remove_auto_ann.sh [ROOT_DIR]
# ROOT_DIR defaults to /data/eeg/scalp/ltp

ROOT="${1:-/data/eeg/scalp/ltp}"

find "$ROOT" -type d | while read -r dir; do
    if ls "$dir"/whisperx* &>/dev/null 2>&1; then
        ann_files=("$dir"/*.ann)
        if [ -e "${ann_files[0]}" ]; then
            echo "Removing .ann files in: $dir"
            rm "${ann_files[@]}"
        fi
    fi
done
