#!/usr/bin/env python3

from imageanalysisplatesolve import findfilesindir
from pathlib import Path
from astropy.io import fits
from reproject import reproject_interp
from ccdproc import Combiner, CCDData

import numpy as np

def groupfiles(allfns):
    result = {}
    for fn in allfns:
        with fits.open(fn) as hdul:
            header = hdul[0].header
            target = None
            for key in ["OBJNAME","OBJECT"]:
                try:
                    target = header[key]
                except KeyError:
                    pass
                else:
                    break
            telescope = None
            for key in ["TELESCOP","iTelescope"]:
                try:
                    telescope = header[key]
                except KeyError:
                    pass
                else:
                    break
            filtername = None
            for key in ["FILTER"]:
                try:
                    filtername = header[key]
                except KeyError:
                    pass
                else:
                    break
            exposure = None
            for key in ["EXPOSURE","EXPTIME","XPOSURE"]:
                try:
                    exposure = header[key]
                except KeyError:
                    pass
                else:
                    break
            xbinning = None
            ybinning = None
            try:
                xbinning = header["XBINNING"]
            except KeyError:
                pass
            try:
                ybinning = header["YBINNING"]
            except KeyError:
                pass
            if xbinning != ybinning:
                raise Exception("xbinning != ybinning")
            binning = xbinning
            #print(f"target: {target} telescope: {telescope} binning: {binning} filter: {filtername} exposure: {exposure}")
            if False:
                for key in set(header):
                    print(key,header[key])
                return []
            key = f"{target}-{telescope.replace(' ','_')}-bin{binning}-{filtername}-{exposure:.0f}.fit"
            imgdata = {
                "target": target,
                "telescope": target,
                "binning": binning,
                "filter": filtername,
                "exposure": exposure,
            }
            try:
                result[key][0].append(fn)
            except KeyError:
                result[key] = ([fn],imgdata)
    for key in result:
        print(key)
        print(result[key][1])
        for fn in result[key][0]:
            print("   ",fn)
    return result

def stack(fnlist,outfn,outdir):
    print(f"Stacking {outfn} ...")
    reprojectdir = outdir / "reprojected"
    reprojectdir.mkdir(exist_ok=True)
    #ccddata = CCDData.read(fn,unit="adu")
    if len(fnlist) < 2:
        NotImplementedError("stack requires 2 or more files")
    reffn = fnlist[0]
    reproject_fns = []
    with fits.open(reffn) as refhdul:
        refhdu = refhdul[0]
        for fn in fnlist[1:]:
            print(f"Reprojecting {fn} to reference {reffn} ...")
            with fits.open(fn) as hdul:
                hdu = hdul[0]
                reproject_array, reproject_footprint = reproject_interp(hdu,refhdu.header)
                reproject_fn = reprojectdir / Path(fn.stem + ".fit")
                reproject_header = fits.Header({"RPRJTO":str(reffn)})
                fits.writeto(reproject_fn,reproject_array,reproject_header,overwrite=True)
                reproject_fns.append(reproject_fn)
    ccddatas = [CCDData.read(fn,unit="adu") for fn in [reffn]+reproject_fns]
    print(f"Combining images...")
    combiner = Combiner(ccddatas)
    combiner.sigma_clipping()
    avgimg = combiner.average_combine()
    avgimg.write(outdir / outfn)
    print(f"Wrote out {outdir / outfn}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        prog="imagestack.py",
        description="Stacks all images in the input directories that share target, telescope, binning, filter, and exposure time"
    )
    parser.add_argument("indir",type=Path,nargs="+",help="Input directories to search for *.fit and *.fit.zip files")
    parser.add_argument("outdir",type=Path,help="Directory where output files will be written")

    args = parser.parse_args()

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
        stack(groups[key][0],key,outdir)

if __name__ == "__main__":
    main()
