# Provide path to processed pdfs. 
ACADEMIC_OUT="/workspace/datasets/pdfs_txt/"

for fin in $ACADEMIC_OUT*; do
    if [[ $(wc -w < "$fin") -lt 250 ]]; then
        if [[ $(python3 /workspace/chipgpt/pdfcleaner/count_words.py < "$fin") -lt 25 ]]; then
            rm "$fin"
        fi
    fi
done