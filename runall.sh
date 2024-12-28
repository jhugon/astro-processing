#!/bin/bash

pipenv run python imagecalibrate.py darks ~/astrophotography/at65edq-asi533mcpro/*/Autorun/Dark/ masterdarks/
pipenv run python imagecalibrate.py lights --darks masterdarks/ ~/astrophotography/at65edq-asi533mcpro/*/Plan/Light/ ~/astrophotography/at65edq-asi533mcpro/*/Autorun/Light/ calibrated
pipenv run python imageanalysisplatesolve -l 60 calibrated/ analyzed/
