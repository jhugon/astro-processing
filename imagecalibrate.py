#!/usr/bin/env python3

from imageanalysisplatesolve import findfilesindir, checkiffileneedsupdate
from imagestack import groupfiles
from pathlib import Path
from astropy.io import fits
import astropy.units as u
from ccdproc import Combiner, CCDData, subtract_dark, flat_correct
from astropy.wcs import FITSFixedWarning

import sys
import numpy as np
import cv2 as cv


def get_headers(headers):
    result = {}
    for key in ["exposure","ccd-temp","gain","xbinning","ybinning","set-temp","imagetyp","date-obs","bayerpat","egain"]:
        result[key] = headers[key]
    return result

def make_key(headers):
    exposure = headers["exposure"]
    temp = headers["set-temp"]
    gain = headers["gain"]
    xbinning = headers["xbinning"]
    ybinning = headers["ybinning"]
    key = f"gain{gain}-xbin{xbinning}-ybin{ybinning}-temp{temp:.0f}-exp{exposure:.0f}"
    return key

def groupfiles(allfns):
    result = {}
    for fn in allfns:
        with fits.open(fn) as hdul:
            header = hdul[0].header
            key = make_key(header)
            try:
                result[key][0].append(fn)
            except KeyError:
                result[key] = ([fn],get_headers(header))
    for key in sorted(result):
        print(key)
        print(result[key][1])
        for fn in result[key][0]:
            print("   ",fn)
    return result

def get_dark(key,darks):
    for dark in darks:
        ccd = CCDData.read(dark,unit="adu")
        if make_key(ccd.header) == key:
            return ccd
    raise Exception(f"Dark not found for light key: {key}")

def _get_bayer_code(ccd):
    bayerpat = ccd.header["bayerpat"].strip()
    match bayerpat:
        case "RGGB":
            #return cv.COLOR_BayerRGGB2RGB
            return cv.COLOR_BayerRGGB2GRAY
        case "BGGR":
            #return cv.COLOR_BayerBGGR2RGB
            return cv.COLOR_BayerBGGR2GRAY
        case _:
            ValueError(f"Bayerpat string '{bayerpat}' not recognized")

def demosaic(ccd):
    if ccd.header["bayerpat"]:
        resultdata = cv.demosaicing(ccd.data,_get_bayer_code(ccd))
        result = CCDData(resultdata,unit=u.adu,header=ccd.header)
        result.header.pop("bayerpat")
        return result
    else:
        return ccd

def calibratelights(args):

    indirs = args.indir
    outdir = args.outdir

    for indir in indirs:
        if not indir.exists():
            raise Exception(f"{indir} doesn't exist")
        if not indir.is_dir():
            raise Exception(f"{indir} isn't a directory")
    darkfns = None
    if args.darks:
        if not args.darks.exists():
            raise Exception(f"{args.darks} doesn't exist")
        if not args.darks.is_dir():
            raise Exception(f"{args.darks} isn't a directory")
        darkfns = findfilesindir(args.darks)
    if args.flats:
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

    for fn in infiles:
        outfn = Path(str(outdir / fn.stem) + ".fit")
        if not checkiffileneedsupdate([fn],outfn):
            continue
        ccd = CCDData.read(fn,unit="adu")
        if ccd.header["bayerpat"]:
            ccd = demosaic(ccd)
        exposure = ccd.header["EXPOSURE"]
        temp = ccd.header["SET-TEMP"]
        gain = ccd.header["GAIN"]
        print(f"Calibrating {fn} {exposure} s {temp} C {gain} gain...")
        if darkfns:
            light_key = make_key(ccd.header)
            try:
                dark = get_dark(light_key,darkfns)
            except Exception as e:
                print(f"Error: {e}",file=sys.stderr)
                continue
            ccd = subtract_dark(ccd,dark,exposure_time="EXPOSURE",exposure_unit=u.second)
        if args.flats:
            raise NotImplementedError()
            #ccd = flat_correct(ccd,flat)
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
    for key in groups:
        outfnbase = "masterdark-" + key + ".fit"
        outfn = outdir / outfnbase
        print(f"Stacking {outfn} ...")
        fns, info = groups[key]
        combiner = Combiner(map(lambda fn: demosaic(CCDData.read(fn,unit="adu")),fns))
        combiner.sigma_clipping()
        masterdark = combiner.average_combine()
        for key in info:
            masterdark.header[key] = info[key]
        masterdark.write(outdir / outfnbase,overwrite=True)

def stackflats(fnlist,outdir):
    raise NotImplementedError()

def main():
    from astropy import log
    import argparse
    import warnings
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
    parser_lights.add_argument("--darks",type=Path,help="Directory to find master darks")
    parser_lights.add_argument("--flats",type=Path,help="Directory to find master flats")

    parser_darks.add_argument("indir",type=Path,nargs="+",help="Input directories to search for *.fit and *.fit.zip files")
    parser_darks.add_argument("outdir",type=Path,help="Directory where output files will be written")

    warnings.simplefilter("ignore", FITSFixedWarning)
    log.setLevel("WARNING")

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
