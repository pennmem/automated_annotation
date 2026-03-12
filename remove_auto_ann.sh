#!/bin/bash
# Remove auto-generated .ann files from session directories that contain whisperx* files.
# Also removes the corresponding .ann files from the parse_files SVN working copy and commits.
#
# Usage: ./remove_auto_ann.sh [ROOT_DIR] [PARSE_FILES_DIR]
#   ROOT_DIR        defaults to /data/eeg/scalp/ltp
#   PARSE_FILES_DIR defaults to ~/parse_files

ROOT="${1:-/data/eeg/scalp/ltp}"
PARSE_FILES_DIR="${2:-$HOME/parse_files}"

removed_pf=()

find "$ROOT" -type d | while read -r dir; do
    if ls "$dir"/whisperx* &>/dev/null 2>&1; then
        ann_files=("$dir"/*.ann)
        if [ -e "${ann_files[0]}" ]; then
            echo "Removing .ann files in: $dir"
            rm "${ann_files[@]}"

            # Mirror removal into parse_files SVN working copy
            rel="$(realpath --relative-to="$ROOT" "$dir")"
            pf_dir="$PARSE_FILES_DIR/$rel"
            if [ -d "$pf_dir" ]; then
                pf_ann_files=("$pf_dir"/*.ann)
                if [ -e "${pf_ann_files[0]}" ]; then
                    echo "  svn delete .ann files in: $pf_dir"
                    svn delete "${pf_ann_files[@]}"
                fi
            fi
        fi
    fi
done

# Commit all deletions in one pass
if svn status "$PARSE_FILES_DIR" | grep -q '^D'; then
    echo "Committing SVN deletions in $PARSE_FILES_DIR ..."
    # svn commit "$PARSE_FILES_DIR" -m "Remove auto-generated .ann files (whisperx* present)"
    echo "SVN commit done."
else
    echo "No SVN deletions to commit."
fi
