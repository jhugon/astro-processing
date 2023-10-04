#!/usr/bin/env python3

from pathlib import Path
import re
import requests_cache

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.visualization import simple_norm
from ccdproc import CCDData
from photutils.background import Background2D
from photutils.detection import find_peaks
from photutils.aperture import CircularAperture, SkyCircularAperture
from photutils.aperture import ApertureStats
from astroquery.vizier import Vizier

from imageanalysisplatesolve import findfilesindir, checkiffileneedsupdate

def parse_target(fn: Path):
    m = re.match(r"calibrated-([tT]\d+)-\w+-(\w+)-\d+-\d+-(\w+)-BIN\d+-\w-\d+-\d+.fit",fn.name)
    if m:
        return m.group(2)
    else:
        raise ValueError(f"Can't parse filename: {fn.name}")

def load_vsp(ra,dec,std_field=False,session=None):
    url = f"https://app.aavso.org/vsp/api/chart/"
    params = {
        "ra": ra,
        "dec": dec,
        "fov": 40,
        "maglimit": 16.5,
    }
    headers = {
        "Accept": "application/json",
    }
    if std_field:
        params["special"] = "std_field"
    response = session.get(url,params=params,headers=headers)
    response.raise_for_status()
    data = response.json()
    photometry = data["photometry"]
    return photometry
    
def analyze(fn,outdir,session):
    print(f"Analyzing {fn} ...")
    outfile = ( outdir / fn.stem ).with_suffix( ".fit")
    if not checkiffileneedsupdate([fn],outfile):
        print(f"No update needed for output file {outfile}")
        return
    target = parse_target(fn)
    with fits.open(fn) as hdul:
        hdu = hdul[0]
        fwhmpx = hdu.header['FWHMPX']
        bkmean = hdu.header["BKMEAN"]
        bkstd = hdu.header["BKSTD"]
        print(f"FWHM: {fwhmpx} pix, bkg: {bkmean}, std: {bkstd}")
        wcs = WCS(hdu.header)
        vsp_data = load_vsp(hdu.header["RA"],hdu.header["DEC"],True,session)
        skypos = SkyCoord(ra=[line["ra"] for line in vsp_data],dec=[line["dec"] for line in vsp_data],unit=(u.hourangle,u.deg))
        pixpos = skypos.to_pixel(wcs)
        skyaperture = SkyCircularAperture(skypos,r=15*u.arcsec)
        pixaperture = CircularAperture(zip(*pixpos),r=int(fwhmpx*2))
        aperstats = ApertureStats(hdu.data, pixaperture)

        norm = simple_norm(hdu.data,'sqrt',percent=99)
        plt.imshow(hdu.data,norm=norm,interpolation="nearest")
        ap_patches = pixaperture.plot(color='white',lw=2)
        plt.show()


def main():

    import argparse
    parser = argparse.ArgumentParser(
        prog="imageanalysisplatesolve.py",
        description="Creates new copies of images with fits headers including WCS, FWHM, background, noise, and S/N. Standard WCS fits headers are added as well as BKMEAN, BKMEDIAN, and BKSTD, which are the 3-sigma-clipped mean, median, and standard-deviation of the image."
    )
    parser.add_argument("indir",type=Path,nargs="+",help="Input directories to search for *.fit and *.fit.zip files.")
    parser.add_argument("outdir",type=Path,help="Directory where output files will be written.")

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

    session = requests_cache.CachedSession()
    for infile in infiles:
        analyze(infile,outdir,session)
        

if __name__ == "__main__":
    main()
