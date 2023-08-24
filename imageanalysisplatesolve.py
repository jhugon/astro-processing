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

def checkiffileneedsupdate(infiles: [Path],outfile: Path) -> bool:
    if not outfile.exists():
        return True
    outfilemtime = outfile.stat().st_mtime
    infilemtime = float('-inf')
    for infile in infiles:
        infilestat = infile.stat()
        if infilestat.st_mtime > infilemtime:
            infilemtime = infilestat.st_mtime
    if infilemtime > outfilemtime:
        return True
    return False


def runastrometrydotnet(fn,exedir,header,astrometrytimeout):
    print("Running solve-field ...")
    command = ["solve-field","--scale-units","arcsecperpix","--scale-low", "0.1","--guess-scale","--no-plots","-D",exedir,"-l",f"{astrometrytimeout:.0f}",fn]
    subprocess.run(command,cwd=exedir,check=True)
    result = fn.with_suffix(".new")
    if result.exists():
        return result
    else:
        raise Exception(f"solve-field did not solve (or no WCS file was written)")


def analyze(fn,outdir,astrometrytimeout):
    print(f"Analyzing {fn} ...")
    iszipped = fn.suffix == ".zip"
    outfile = ( outdir / fn.stem ).with_suffix( ".fit")
    if not checkiffileneedsupdate([fn],outfile):
        print(f"No update needed for output file {outfile}")
        return
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
        header = None
        with fits.open(tmpfn) as hdul:
            hdu = hdul[0]
            header = hdu.header.copy()
        try:
            adnfn = runastrometrydotnet(tmpfn,tempdir,header,astrometrytimeout)
        except Exception as e:
            print(f"Error: {e}, skipping file.")
            return
        with fits.open(adnfn,mode="update") as hdul:
            hdu = hdul[0]
            print("Analyzing background ...")
            mean, median, std = sigma_clipped_stats(hdu.data,sigma=3.0)
            print(f"Mean Background:   {mean:.1f} ADU")
            print(f"Median Background: {median:.1f} ADU")
            print(f"Noise:             {std:.1f} ADU")
            hdu.header["BKMEAN"] = mean
            hdu.header["BKMEDIAN"] = median
            hdu.header["BKSTD"] = median
        copyfile(adnfn,outfile)


def main():
    DEFAULTTIMEOUT=300.

    import argparse
    parser = argparse.ArgumentParser(
        prog="imageanalysisplatesolve.py",
        description="Creates new copies of images with fits headers including WCS, FWHM, background, noise, and S/N. Standard WCS fits headers are added as well as BKMEAN, BKMEDIAN, and BKSTD, which are the 3-sigma-clipped mean, median, and standard-deviation of the image."
    )
    parser.add_argument("indir",type=Path,nargs="+",help="Input directories to search for *.fit and *.fit.zip files.")
    parser.add_argument("outdir",type=Path,help="Directory where output files will be written.")
    parser.add_argument("--astrometry-timeout",'-l',type=float,help=f"Timeout for astrometry for each image. Default: {DEFAULTTIMEOUT:.0f} s. Timed-out images are skipped.",default=DEFAULTTIMEOUT)

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
        analyze(infile,outdir,args.astrometry_timeout)
    
        

if __name__ == "__main__":
    main()
