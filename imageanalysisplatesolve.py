#!/usr/bin/env python3

from pathlib import Path

from astropy.stats import sigma_clipped_stats
from ccdproc import CCDData
from photutils.background import Background2D

def main():
    import argparse
    parser = argparse.ArgumentParser(
        prog="imageanalysisplatesolve.py",
        description="Creates new copies of images with fits headers including WCS, FWHM, background, noise, and S/N"
    )
    parser.add_argument("indir",type=Path,nargs="+",help="Input directories to search for *.fit and *.fit.zip files")
    parser.add_argument("outdir",type=Path,help="Directory where output files will be written")

    args = parser.parse_args()

    indirs = args.indir
    outdir = args.outdir

    for indir in indirs:
        if not indir.exists():
            raise IOError(f"{indir} doesn't exist")
        if not indir.is_dir():
            raise IOError(f"{indir} isn't a directory")
        print(indir)
    if not outdir.exists():
        outdir.mkdir(parents=True)
    if not outdir.is_dir():
        raise IOError(f"{outdir} isn't a directory")
        

if __name__ == "__main__":
    main()
