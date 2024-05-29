ACADEMIC_IN="/workspace/datasets/pdfs/"
ACADEMIC_OUT="/workspace/datasets/pdfs_txt/"
MANUALS_IN="/all_tool_pdfs/"
MANUALS_OUT="/all_tool_txt/"

python3 convert.py --path $ACADEMIC_IN --outpath $ACADEMIC_OUT --filter academic
python3 convert.py --path $MANUALS_IN --outpath $MANUALS_OUT --filter manuals

# Delete all fins with fewer than 25 non-numerical words
# Only check fins with fewer than 250 total words
for fin in $ACADEMIC_OUT*; do
    if [[ $(wc -w < "$fin") -lt 250 ]]; then
        if [[ $(python3 count_words.py < "$fin") -lt 25 ]]; then
            rm "$fin"
        fi
    fi
done

for fin in $MANUALS_OUT*; do
    if [[ $(wc -w < "$fin") -lt 250 ]]; then
        if [[ $(python3 count_words.py < "$fin") -lt 25 ]]; then
            rm "$fin"
        fi
    fi
done
