#!/usr/bin/env python3

from imageanalysisplatesolve import findfilesindir, checkiffileneedsupdate
from imagestack import groupfiles
from pathlib import Path
from astropy.io import fits
from ccdproc import Combiner, CCDData, subtract_dark, flat_correct

import sys
import numpy as np

def make_key(headers):
    result = {}
    for key in ["exposure","ccd-temp","gain"]:
        result[key] = headers[key]
    return result

def get_dark(key,darks):
    print(f"light key: {key}")
    for dark in darks:
        ccd = CCDData.read(dark,unit="adu")
        print(f"dark key: {make_key(ccd.header)}")
        if make_key(ccd.header) == key:
            return ccd
    raise Exception(f"Dark not found.")

def calibrate(fnlist,outdir,darks,flats):
    for fn in fnlist:
        ccd = CCDData.read(fn,unit="adu")
        exposure = ccd.header["EXPOSURE"]
        temp = ccd.header["SET-TEMP"]
        gain = ccd.header["GAIN"]
        print(f"Calibrating {fn} {exposure} s {temp} C {gain} gain...")
        light_key = make_key(ccd.header)
        dark = get_dark(light_key,darks)
        dark_sub_ccd = subtract_dark(ccd,dark)
        corr_ccd = flat_correct(ccd,flat)
        corr_ccd.write(outdir / fn)

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
