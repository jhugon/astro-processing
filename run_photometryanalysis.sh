#!/bin/bash

command=photometryanalysis.py
command=photometryanalysisnocolor.py

pipenv run python $command T21-photometry/calibrated-T21-jhugon-AT_And-* --target "AT And" --comp 000-BCR-873
pipenv run python $command T21-photometry/calibrated-T21-jhugon-AC_And-* --target "AC And" --comp 000-BJR-367
pipenv run python $command T21-photometry/calibrated-T21-jhugon-SW_And-* --target "SW And" --comp 000-BBB-588
