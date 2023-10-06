#!/usr/bin/env python3

from pathlib import Path
import re
import requests_cache

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.table import Table
from astropy import units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.visualization import simple_norm
from ccdproc import CCDData
from photutils.background import Background2D
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, SkyCircularAperture, CircularAnnulus
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
    
def analyze(fn,session):
    print(f"Analyzing {fn} ...")
    target = parse_target(fn)
    with fits.open(fn) as hdul:
        image = hdul[0]
        wcs = WCS(image.header)

        sources = Table.read(hdul[1])
        phottable = Table.read(hdul[2])
        photbktable = Table.read(hdul[3])

        breakpoint()

        vsp_data = load_vsp(image.header["RA"],image.header["DEC"],True,session)
        #skypos = SkyCoord(ra=[line["ra"] for line in vsp_data],dec=[line["dec"] for line in vsp_data],unit=(u.hourangle,u.deg))
        #pixpos = skypos.to_pixel(wcs)
        #skyaperture = SkyCircularAperture(skypos,r=15*u.arcsec)
        #pixaperture = CircularAperture(zip(*pixpos),r=int(fwhmpx*2))
        #pixaperstats = ApertureStats(hdu.data, pixaperture)
        ##pixap_patches = pixaperture.plot(color='red')

        #norm = simple_norm(hdu.data,'sqrt',percent=99)
        #plt.imshow(hdu.data,norm=norm,interpolation="nearest")
        #ap_patches = apertures.plot(color='white')
        #an_patches = annuluses.plot(color='white')
        #plt.show()


def main():

    import argparse
    parser = argparse.ArgumentParser(
        prog="imageimvestigatephotometry.py",
        description="Investiage photometry script output."
    )
    parser.add_argument("infile",type=Path,nargs="+",help="Input fits file(s)")

    args = parser.parse_args()

    session = requests_cache.CachedSession()
    for infile in args.infile:
        analyze(infile,session)
        

if __name__ == "__main__":
    main()
