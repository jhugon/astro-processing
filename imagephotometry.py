#!/usr/bin/env python3

from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.table import QTable
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

def analyze(fn,outdir):
    print(f"Analyzing {fn} ...")
    outfile = ( outdir / fn.stem ).with_suffix( ".fit")
    if not checkiffileneedsupdate([fn],outfile):
        print(f"No update needed for output file {outfile}")
        return
    with fits.open(fn) as hdul:
        hdu = hdul[0]
        fwhmpx = hdu.header['FWHMPX']
        bkmean = hdu.header["BKMEAN"]
        bkstd = hdu.header["BKSTD"]
        exposure = float(hdu.header["EXPOSURE"])
        print(f"FWHM: {fwhmpx} pix, bkg: {bkmean}, std: {bkstd}")
        wcs = WCS(hdu.header)
        #bkg2d = Background2D(hdu.data, 64, filter_size = 3)

        daofind = DAOStarFinder(fwhm=fwhmpx, threshold = 5 * bkstd)
        sources = daofind(hdu.data - bkmean)
        positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
        skypositions = SkyCoord.from_pixel(sources["xcentroid"],sources["ycentroid"],wcs)
        aperture_R = fwhmpx*1.75
        annulus_Rin = fwhmpx*5
        annulus_Rout = fwhmpx*9
        annuluses = CircularAnnulus(positions, r_in=annulus_Rin, r_out=annulus_Rout)
        apertures = CircularAperture(positions, r=aperture_R)

        bkgstats = ApertureStats(hdu.data,annuluses,sigma_clip=SigmaClip(sigma=3.0,maxiters=10))
        aperstats = ApertureStats(hdu.data, apertures,sigma_clip=None)

        flux = aperstats.sum
        flux_bkg = bkgstats.mean * aperstats.sum_aper_area.value
        flux_bkg_sub = flux-flux_bkg
        instmag = -2.5 * np.log10(flux_bkg_sub/exposure)
        
        phottable = QTable([instmag,flux_bkg_sub,flux,flux_bkg,sources["xcentroid"],sources["ycentroid"],skypositions.ra.to_string("deg"),skypositions.dec.to_string("deg")],
                            names=("Instrumental Magnitude","Flux - Background", "Flux", "Background Flux","x","y","ra","dec"),
                            meta={"name": "PHOTTABLE","R":aperture_R,"RIN": annulus_Rin, "ROUT": annulus_Rout}
                        )

        # Output file

        hdul_out = fits.HDUList()
        aperturetable = aperstats.to_table()
        annulustable = bkgstats.to_table()
        del aperturetable["sky_centroid"]
        del annulustable["sky_centroid"]
        sources_hdu = fits.BinTableHDU(sources,name="SOURCES")
        aperturetable_header = fits.Header()
        aperturetable_header["R"] = aperture_R
        annulustable_header = fits.Header()
        annulustable_header["RIN"] = annulus_Rin
        annulustable_header["ROUT"] = annulus_Rout
        phottable_hdu = fits.BinTableHDU(phottable,name="PHOT")
        aperturetable_hdu = fits.BinTableHDU(aperturetable,name="APERTURE",header=aperturetable_header)
        annulustable_hdu = fits.BinTableHDU(annulustable,name="ANNULUS",header=annulustable_header)
        hdul_out.append(hdu)
        hdul_out.append(phottable_hdu)
        hdul_out.append(sources_hdu)
        hdul_out.append(aperturetable_hdu)
        hdul_out.append(annulustable_hdu)
        hdul_out.writeto(outfile,overwrite=True)

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

    for infile in infiles:
        analyze(infile,outdir)
        

if __name__ == "__main__":
    main()
