This directory contains the data collection scripts.


Currently requires Python 3.10

First add /home/utils/Python-3.10/bin to PATH.

Then run `python3 -m virtualenv venv` at top of repo.

Then `source venv/bin/activate.csh`

Finally, install prereqs with `pip install -r requirements.txt`. 

* Playbook for data curation with NeMo Curator: `\notebooks\data_curation-nemo-curator_DAPT.ipynb`
* Playbook for data curation without NeMo Curator: `\notebooks\data_curation_DAPT.ipynb`
* Playbook for data curation of pdfs (with conversion to txt): `\notebooks\data_curation_pdf.ipynb`


