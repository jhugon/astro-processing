#!/usr/bin/env python3

from imageanalysisplatesolve import findfilesindir, checkiffileneedsupdate
from imagestack import groupfiles
from pathlib import Path
from astropy.io import fits
from ccdproc import Combiner, CCDData

import sys
import numpy as np

def calibrate(fnlist,outdir,darks,flats):
    groups = groupfiles(fnlist)
    for key in sorted(groups):
        fns, imgdata = groups[key]
        print(imgdata)
        print(fns)

def stackdarks(fnlist,outdir):
    pass

def stackflats(fnlist,outdir):
    pass

def main():
    import argparse
    parser = argparse.ArgumentParser(
        prog="imagecalibrate.py",
        description="Calibrates light frames by subtracting darks, and dividing by flats. Also stacks darks and flats when using the appropriate options."
    )
    parser.add_argument("--stackdarks",action="store_true",help="Stacks dark frames instead of calibrating light frames")
    parser.add_argument("--stackflats",action="store_true",help="Alternatively stacks dark frames")
    parser.add_argument("indir",type=Path,nargs="+",help="Input directories to search for *.fit and *.fit.zip files")
    parser.add_argument("outdir",type=Path,help="Directory where output files will be written")
    parser.add_argument("darks",type=Path,help="Directory to search for *.fit and *.fit.zip files for darks")
    parser.add_argument("flats",type=Path,help="Directory to search for *.fit and *.fit.zip files for flats")

    args = parser.parse_args()

    indirs = args.indir
    outdir = args.outdir

    for indir in indirs:
        if not indir.exists():
            raise Exception(f"{indir} doesn't exist")
        if not indir.is_dir():
            raise Exception(f"{indir} isn't a directory")
    if not args.darks.exists():
        raise Exception(f"{args.darks} doesn't exist")
    if not args.darks.is_dir():
        raise Exception(f"{args.darks} isn't a directory")
    if not args.flats.exists():
        raise Exception(f"{args.flats} doesn't exist")
    if not args.flats.is_dir():
        raise Exception(f"{args.flats} isn't a directory")
    if not outdir.exists():
        outdir.mkdir(parents=True)
    if not outdir.is_dir():
        raise IOError(f"{outdir} isn't a directory")

    infiles = []
    for indir in indirs:
        infiles += findfilesindir(indir)

    darks = findfilesindir(args.darks)
    flats = findfilesindir(args.flats)

    if args.stackdarks and args.stackflats:
        print("Error: choose only one of --stackdarks and --stackflats",file=sys.stderr)
        sys.exit(1)
    elif args.stackdarks:
        stackdarks(infiles,outdir)
    elif args.stackflats:
        stackflats(infiles,outdir)
    else:
        calibrate(infiles,outdir,darks,flats)

if __name__ == "__main__":
    main()
