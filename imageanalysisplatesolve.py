#!/usr/bin/env python3

from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from ccdproc import CCDData
from photutils.background import Background2D
from photutils.detection import find_peaks
from shutil import copyfile

def findfilesindir(p):
    if p.is_file():
        if p.suffix == ".fit":
            return [p]
        elif len(p.suffixes) > 1 and p.suffixes[-2] == ".fit" and p.suffix in [".zip",".gz"]:
            return [p]
        else:
            return []
    elif p.is_dir():
        result = []
        for p2 in p.iterdir():
            tmp = findfilesindir(p2)
            result += tmp
        return result
    else:
        raise Exception(f"{p} isn't a file or directory")

def runastrometrydotnet(fn,exedir):
    print("Running astrometry.net ...")
    command = ["solve-field",fn]
    subprocess.run(command,cwd=exedir,check=True)
    return fn.with_suffix(".new")

def analyze(fn,outdir):
    print(f"Analyzing {fn} ...")
    with TemporaryDirectory() as tempdir:
        tmpfn = fn
        if fn.suffix == ".zip":
            try:
                subprocess.run(["unzip",str(fn)],cwd=tempdir,check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error while running unzip: {e}, skipping file.")
                return
            else:
                tmpfn = Path(tempdir) / Path(fn.stem)
        adnfn = runastrometrydotnet(tmpfn,tempdir)
        with fits.open(adnfn,mode="update") as hdul:
            hdu = hdul[0]
            print("Analyzing background ...")
            mean, median, std = sigma_clipped_stats(hdu.data,sigma=3.0)
            print(f"Mean Background:   {mean:.1f} ADU")
            print(f"Median Background: {median:.1f} ADU")
            print(f"Noise:             {std:.1f} ADU")
        outfile = ( outdir / tmpfn.stem ).with_suffix( ".fit")
        copyfile(adnfn,outfile)

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
    for infile in infiles:
        analyze(infile,outdir)
    
        

if __name__ == "__main__":
    main()
