Docker container is at `/docker/`

This flow uses libpoppler v23.04

Edit the paths in run.sh then run:
```
pip install -e .
bash run.sh
```

Current capabilities:
 - Parse PDFs
 - Iterate over parsed text in terms of documents > pages > flows > blocks > lines

Capabilities that can be added:
 - Bring in a table parser
 - Bring in an image extractor
