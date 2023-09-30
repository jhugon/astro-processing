#!/usr/bin/env python3

from imageanalysisplatesolve import findfilesindir, checkiffileneedsupdate
from imagestack import groupfiles
from pathlib import Path
from astropy.io import fits
import astropy.units as u
from ccdproc import Combiner, CCDData, subtract_dark, flat_correct

import sys
import numpy as np

def make_key(headers):
    result = {}
    for key in ["exposure","ccd-temp","gain","xbinning","ybinning","set-temp","imagetyp","date-obs","bayerpat","egain"]:
        result[key] = headers[key]
    if result["xbinning"] != result["ybinning"]:
        raise Exception("xbinning != ybinning")
    return result

def groupfiles(allfns):
    result = {}
    for fn in allfns:
        with fits.open(fn) as hdul:
            header = hdul[0].header
            keydict = make_key(header)
            exposure = keydict["exposure"]
            ccdtemp = keydict["ccd-temp"]
            gain = keydict["gain"]
            xbinning = keydict["xbinning"]
            ybinning = keydict["ybinning"]
            key = f"masterdark-gain{gain}-xbin{xbinning}-ybin{ybinning}-temp{ccdtemp:.0f}-exp{exposure:.0f}.fit"
            try:
                result[key][0].append(fn)
            except KeyError:
                result[key] = ([fn],keydict)
    for key in sorted(result):
        print(key)
        print(result[key][1])
        for fn in result[key][0]:
            print("   ",fn)
    return result

def get_dark(key,darks):
    print(f"light key: {key}")
    for dark in darks:
        ccd = CCDData.read(dark,unit="adu")
        print(f"dark key: {make_key(ccd.header)}")
        if make_key(ccd.header) == key:
            return ccd
    raise Exception(f"Dark not found.")

def calibratelights(args):

    indirs = args.indir
    outdir = args.outdir

    for indir in indirs:
        if not indir.exists():
            raise Exception(f"{indir} doesn't exist")
        if not indir.is_dir():
            raise Exception(f"{indir} isn't a directory")
    #if not args.darks.exists():
    #    raise Exception(f"{args.darks} doesn't exist")
    #if not args.darks.is_dir():
    #    raise Exception(f"{args.darks} isn't a directory")
    #if not args.flats.exists():
    #    raise Exception(f"{args.flats} doesn't exist")
    #if not args.flats.is_dir():
    #    raise Exception(f"{args.flats} isn't a directory")
    if not outdir.exists():
        outdir.mkdir(parents=True)
    if not outdir.is_dir():
        raise IOError(f"{outdir} isn't a directory")

    infiles = []
    for indir in indirs:
        infiles += findfilesindir(indir)

    dark = args.dark
    flat = args.flat

    for fn in infiles:
        outfn = Path(str(outdir / fn.stem) + ".fit")
        ccd = CCDData.read(fn,unit="adu")
        exposure = ccd.header["EXPOSURE"]
        temp = ccd.header["SET-TEMP"]
        gain = ccd.header["GAIN"]
        print(f"Calibrating {fn} {exposure} s {temp} C {gain} gain...")
        #light_key = make_key(ccd.header)
        #dark = get_dark(light_key,darks)
        if args.dark:
            dark = CCDData.read(args.dark,unit="adu")
            ccd = subtract_dark(ccd,dark,exposure_time="EXPOSURE",exposure_unit=u.second)
        if args.flat:
            flat = CCDData.read(args.flat,unit="adu")
            ccd = flat_correct(ccd,flat)
        ccd.write(outfn,overwrite=True)

def stackdarks(args):
    indirs = args.indir
    outdir = args.outdir

    for indir in indirs:
        if not indir.exists():
            raise Exception(f"{indir} doesn't exist")
        if not indir.is_dir():
            raise Exception(f"{indir} isn't a directory")
    if not outdir.exists():
        outdir.mkdir(parents=True)
    if not outdir.is_dir():
        raise IOError(f"{outdir} isn't a directory")

    infiles = []
    for indir in indirs:
        infiles += findfilesindir(indir)
    groups = groupfiles(infiles)
    for outfnbase in groups:
        outfn = outdir / outfnbase
        print(f"Stacking {outfn} ...")
        fns, info = groups[outfnbase]
        combiner = Combiner(map(lambda x: CCDData.read(x,unit="adu"),fns))
        combiner.sigma_clipping()
        masterdark = combiner.average_combine()
        for key in info:
            masterdark.header[key] = info[key]
        masterdark.write(outdir / outfnbase,overwrite=True)

def stackflats(fnlist,outdir):
    pass

def main():
    import argparse
    parser = argparse.ArgumentParser(
        prog="imagecalibrate.py",
        description="Calibrates light frames by subtracting darks, and dividing by flats. Also stacks darks and flats when using the appropriate options."
    )

    subparsers = parser.add_subparsers()
    parser_lights = subparsers.add_parser("lights",help="Calibrates light frames")
    parser_darks = subparsers.add_parser("darks",help="Stacks dark frames. Overwrites master darks in output directory with combination of whatever is found in input directory. Does 3σ clipping.")
    parser_flats = subparsers.add_parser("flats",help="Stacks and normalizes flat frames. Overwrites master flats in output directory with combination of whatever is found in input directory. Does 3σ clipping.")

    parser_lights.set_defaults(func=calibratelights)
    parser_darks.set_defaults(func=stackdarks)
    parser_flats.set_defaults(func=stackflats)

    parser_lights.add_argument("indir",type=Path,nargs="+",help="Input directories to search for *.fit and *.fit.zip files")
    parser_lights.add_argument("outdir",type=Path,help="Directory where output files will be written")
    parser_lights.add_argument("--dark",type=Path,help="File to use for dark calibration")
    parser_lights.add_argument("--flat",type=Path,help="File to use for flat calibration")

    parser_darks.add_argument("indir",type=Path,nargs="+",help="Input directories to search for *.fit and *.fit.zip files")
    parser_darks.add_argument("outdir",type=Path,help="Directory where output files will be written")

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
