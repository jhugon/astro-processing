#!/usr/bin/env python3

from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
from math import pi as PI
from shutil import copyfile

import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats, SigmaClip
from ccdproc import CCDData
from photutils.background import Background2D
from photutils.detection import find_peaks
from photutils.aperture import CircularAperture
from photutils.aperture import ApertureStats
from astroquery.vizier import Vizier

class SolveFieldError(Exception):
    pass


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

def pixel_scale(header):
    xpixsz = header["XPIXSZ"] # micron
    ypixsz = header["YPIXSZ"]
    fl = header["FOCALLEN"] # mm
    platescale = 1./fl      # radians / mm
    platescale *= 180./PI # deg / mm
    platescale *= 60.*60. # arcsec / mm
    platescale /= 1000. # arcsec / micron
    xpixelscale = platescale * xpixsz
    ypixelscale = platescale * ypixsz
    return xpixelscale, ypixelscale


def runastrometrydotnet(fn,exedir,header,astrometrytimeout,debug=False):
    print("Running solve-field ...")
    command = ["solve-field","--scale-units","arcsecperpix","--guess-scale","--no-plots","-D",exedir,"-l",f"{astrometrytimeout:.0f}",fn]
    if not debug:
        command += ["--no-plots"]
    try:
        ra = header["RA"]
        dec = header["DEC"]
        coord = SkyCoord(ra,dec,unit=(u.hourangle,u.deg))
        command += ["--ra",str(coord.ra.value),"--dec",str(coord.dec.value),"--radius","10"]
    except KeyError:
        pass
    try:
        xpixelscale, ypixelscale = pixel_scale(header)
        minpixelscale = min(xpixelscale,ypixelscale)*0.5
        maxpixelscale = max(xpixelscale,ypixelscale)*2.
        command += ["--scale-low", f"{minpixelscale:.4f}","--scale-high", f"{maxpixelscale:.4f}"]
    except (KeyError,ZeroDivisionError):
        command += ["--scale-low", "0.1"]
    print(command)
    subprocess.run(command,cwd=exedir,check=True)
    result = fn.with_suffix(".new")
    if result.exists():
        return result
    else:
        raise SolveFieldError(f"solve-field did not solve (or no WCS file was written)")

def analyze_fwhm(hdu):
    """
    Writes the result to the input hdu header
    """
    print("Analyzing FWHM...")

    npeaks = 100

    wcs = WCS(hdu.header)
    peaks = find_peaks(hdu.data, threshold=hdu.header["BKMEAN"]+5.0*hdu.header["BKSTD"], box_size=31, npeaks=npeaks,wcs=wcs)
    positions = np.transpose((peaks['x_peak'],peaks['y_peak']))
    apertures = CircularAperture(positions,r=10.0)

    stats = ApertureStats(hdu.data-hdu.header["BKMEAN"],apertures)
    sigmaclip = SigmaClip()
    fwhm_sigclip = sigmaclip(stats.fwhm)
    fwhm_sigclip_mean = fwhm_sigclip.mean()
    hdu.header["FWHMPX"] = (fwhm_sigclip_mean.value,f"[pixels] 3sig mean of {npeaks} brt stars")
    print(f"FWHM: {fwhm_sigclip_mean.value:.2f} pixels")

    #import matplotlib.pyplot as plt
    #from astropy.visualization import SqrtStretch
    #from astropy.visualization.mpl_normalize import ImageNormalize
    #norm = ImageNormalize(stretch=SqrtStretch())
    #plt.imshow(hdu.data, cmap='Greys', origin='lower',norm=norm)
    #apertures.plot(color='blue',lw=1.5,alpha=0.5)
    #plt.show()
    #plt.hist(stats.fwhm,100)
    #plt.axvline(fwhm_sigclip_mean.value)
    #plt.xlim(0,10)
    #plt.show()

    analyze_limiting_mag(hdu,peaks)
    
def analyze_limiting_mag(hdu,peaks):
    catalogs = [
        "II/183A/table2", # Landolt 1992
        "J/AJ/133/2502", # Landolt 2007
        "J/AJ/146/131", # Landolt 2013
    ]
    cats = Vizier.get_catalogs(catalogs)
    #breakpoint()

def analyze(fn,outdir,astrometrytimeout,astrometrydebug):
    print(f"Analyzing {fn} ...")
    fn = fn.absolute()
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
                if not tmpfn.exists():
                    dircontents = [ x for x in Path(tempdir).iterdir()]
                    if len(dircontents) == 1:
                        tmpfn = dircontents[0]
                    else:
                        print(f"Error: directory empty after unzipping {fn}")
                        return
        else:
            tmpfn = Path(tempdir) / "infile.fit"
            tmpfn.symlink_to(Path(fn).absolute())
        header = None
        with fits.open(tmpfn) as hdul:
            hdu = hdul[0]
            header = hdu.header.copy()
        if "BAYERPAT" in header:
            raise ValueError(f"Image '{fn}' should already be demosaiced, but BAYERPAT is in header")
        try:
            adnfn = runastrometrydotnet(tmpfn,tempdir,header,astrometrytimeout)
        except (subprocess.CalledProcessError,SolveFieldError) as e:
            if astrometrydebug:
                print(f"Error: {e}")
                #import os
                #os.environ['XPA_METHOD'] = "local"
                import imexam
                viewer = imexam.connect()
                viewer.load_fits(str(tmpfn))
                viewer.scale()
                viewer.imexam()
                viewer.close()
                breakpoint()
            print(f"Error: {e}, skipping file.")
            return
        else:
            with fits.open(adnfn,mode="update") as hdul:
                hdu = hdul[0]
                print("Analyzing background ...")
                mean, median, std = sigma_clipped_stats(hdu.data,sigma=3.0)
                print(f"Mean Background:   {mean:.1f} ADU")
                print(f"Median Background: {median:.1f} ADU")
                print(f"Noise:             {std:.1f} ADU")
                hdu.header["BKMEAN"] = (mean,"[ADU] 3-sigma clipped")
                hdu.header["BKMEDIAN"] = (median,"[ADU] 3-sigma clipped") 
                hdu.header["BKSTD"] = (std,"[ADU] 3-sigma clipped")
                analyze_fwhm(hdu)
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
    parser.add_argument("--astrometry-debug",action="store_true",help=f"Creates images and pauses if astrometric solution not found")

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
        analyze(infile,outdir,args.astrometry_timeout,args.astrometry_debug)
        

if __name__ == "__main__":
    main()
