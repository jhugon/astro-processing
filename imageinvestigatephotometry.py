#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits, votable
from astropy.table import Table, QTable
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

def analyze(fn):
    print(f"Analyzing {fn} ...")
    with fits.open(fn) as hdul:
        image = hdul[0]
        fwhmpx = image.header['FWHMPX']
        bkmean = image.header["BKMEAN"]
        bkstd = image.header["BKSTD"]
        print(f"FWHM: {fwhmpx} pix, bkg: {bkmean}, std: {bkstd}")

        vsp_table = None
        vsx_table = None
        for hdu in hdul:
            if hdu.name == "VSP":
                vsp_table = QTable.read(hdu)
                vsp_table["imagepos"] = np.transpose((vsp_table["x"],vsp_table["y"]))
                del vsp_table["x"]
                del vsp_table["y"]
                del vsp_table["ra"]
                del vsp_table["dec"]
            elif hdu.name == "VSX":
                try:
                    vsx_table = QTable.read(hdu)
                    vsx_table["imagepos"] = np.transpose((vsx_table["x"],vsx_table["y"]))
                except KeyError:
                    vsx_table = None
                else:
                    del vsx_table["x"]
                    del vsx_table["y"]
                    del vsx_table["ra"]
                    del vsx_table["dec"]
            elif hdu.name == "PHOT":
                phot_table = QTable.read(hdu)

        filtername = image.header["filter"]
        R = phot_table.meta["R"]
        Rin = phot_table.meta["RIN"]
        Rout = phot_table.meta["ROUT"]
        std_field = vsp_table.meta["std_field"]

        if std_field:
            print("This is a standard field")

        print("VSP Stars:")
        print(vsp_table)

        print("VSX Stars:")
        print(vsx_table)

        vsp_apertures = CircularAperture(vsp_table["imagepos"],r=R)
        if vsx_table:
            vsx_apertures = CircularAperture(vsx_table["imagepos"],r=R)

        norm = simple_norm(image.data,'sqrt',percent=99)
        plt.imshow(image.data,norm=norm,interpolation="nearest")
        vsp_ap_patches = vsp_apertures.plot(color='purple',lw=2)
        if vsx_table:
            vsx_ap_patches = vsx_apertures.plot(color='red',lw=2)
        plt.show()

def main():

    import argparse
    parser = argparse.ArgumentParser(
        prog="imageimvestigatephotometry.py",
        description="Investiage photometry script output."
    )
    parser.add_argument("infile",type=Path,nargs="+",help="Input fits file(s)")

    args = parser.parse_args()

    for infile in args.infile:
        analyze(infile)
        

if __name__ == "__main__":
    main()
