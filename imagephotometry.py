#!/usr/bin/env python3

from pathlib import Path
import re
import xml.etree.ElementTree as XMLElementTree
import requests_cache

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
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

def parse_target(fn: Path):
    m = re.match(r"calibrated-([tT]\d+)-\w+-(\w+)-\d+-\d+-(\w+)-BIN\d+-\w-\d+-\d+.fit",fn.name)
    if m:
        return m.group(2)
    else:
        raise ValueError(f"Can't parse filename: {fn.name}")

def load_vsp(ra,dec,std_field=False,session=None,filtername=None):
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
    print(data["chartid"],data["image_uri"])
    photometry = data["photometry"]
    newdata = []
    for star in photometry:
        newrow = {}
        newrow["auid"] = star["auid"]
        newrow["skypos"] = SkyCoord(ra=star["ra"],dec=star["dec"],unit=(u.hourangle,u.deg))
        for band in star["bands"]:
            if filtername and filtername != band["band"]:
                continue
            newrow[band["band"]] = band["mag"] * u.mag
            newrow[band["band"]+"error"] = band["error"] * u.mag
        newdata.append(newrow)
    result = QTable(rows=newdata,meta={"std_field": std_field})
    return result

def load_vsx(ra,dec,session=None):
    url = f"http://www.aavso.org/vsx/index.php"
    params = {
        "view": "query.votable",
        "coords": f"{ra} {dec}",
    }
    response = session.get(url,params=params)
    response.raise_for_status()
    xmltree = XMLElementTree.fromstring(response.text)
    fieldnames = []
    fielddata = []
    for table in xmltree.iter("TABLE"):
        for field in table.iter("FIELD"):
            fieldname = field.attrib["name"]
            fieldnames.append(fieldname)
        for tabledata in table.iter("TABLEDATA"):
            for tr in tabledata.iter("TR"):
                tablerow = []
                for td in tr.iter("TD"):
                    tablerow.append(td.text)
                fielddata.append(tablerow)
    if len(fielddata) == 0:
        return None
    for iRow in range(len(fielddata)):
        for iCol in range(len(fieldnames)):
            if fieldnames[iCol] == "Coords(J2000)":
                ra, dec = fielddata[iRow][iCol].split(',')
                fielddata[iRow][iCol] = SkyCoord(ra=ra,dec=dec,unit=u.deg)
            elif fieldnames[iCol] == "AUID":
                if fielddata[iRow][iCol] is None:
                    fielddata[iRow][iCol] = ""
            try:
                fielddata[iRow][iCol] = float(fielddata[iRow][iCol])
            except ValueError:
                pass
            except TypeError:
                pass
    for iCol in range(len(fieldnames)):
        if fieldnames[iCol] == "Coords(J2000)":
            fieldnames[iCol] = "skypos"
            break
    result = Table(rows=fielddata,names=fieldnames)
    return result

def add_phot_to_vsp_table(phot,vsp):
    idx, d2d, _ = vsp["skypos"].match_to_catalog_sky(phot["skypos"])
    result = vsp.copy()
    del result["skypos"]
    result["Match Distance"] = d2d.to(u.arcsec)
    for colname in ["Instrumental Magnitude","Flux","Background Flux","ra","dec","x","y"]:
        result[colname] = phot[colname][idx]
    return result

def add_phot_to_vsx_table(phot,vsx):
    idx, d2d, _ = vsx["skypos"].match_to_catalog_sky(phot["skypos"])
    result = vsx["AUID","Name"].copy()
    result["Match Distance"] = d2d.to(u.arcsec)
    for colname in ["Instrumental Magnitude","Flux","Background Flux","ra","dec","x","y"]:
        result[colname] = phot[colname][idx]
    return result

def combine_photometry_vsx_vsp(image,photometry,session,std_field):
    photometry["skypos"] = SkyCoord(ra=photometry["ra"],dec=photometry["dec"],unit=u.deg)
    photometry["imagepos"] = np.transpose((photometry["x"],photometry["y"]))
    filtername = image.header["filter"]
    R = photometry.meta["R"]
    Rin = photometry.meta["RIN"]
    Rout = photometry.meta["ROUT"]

    vsp_table = load_vsp(image.header["RA"],image.header["DEC"],std_field,session,filtername=filtername)
    vsx_table = load_vsx(image.header["RA"],image.header["DEC"],session)
    combined_vsp_table = add_phot_to_vsp_table(photometry,vsp_table)
    combined_vsx_table = None
    if vsx_table:
        combined_vsx_table = add_phot_to_vsx_table(photometry,vsx_table)

    #apertures = CircularAperture(photometry["imagepos"],r=R)
    #vsp_apertures = CircularAperture(combined_vsp_table["imagepos"],r=R)
    #if vsx_table:
    #    vsx_apertures = CircularAperture(combined_vsx_table["imagepos"],r=R)
    print("VSP Stars:")
    print(combined_vsp_table)

    print("VSX Stars:")
    print(combined_vsx_table)

    #norm = simple_norm(image.data,'sqrt',percent=99)
    #plt.imshow(image.data,norm=norm,interpolation="nearest")
    #ap_patches = apertures.plot(color='white')
    #vsp_ap_patches = vsp_apertures.plot(color='purple',lw=2)
    #if combined_vsx_table:
    #    vsx_ap_patches = vsx_apertures.plot(color='red',lw=2)
    #plt.show()

    del photometry["skypos"]
    del photometry["imagepos"]

    return combined_vsx_table, combined_vsp_table


def analyze(fn,outdir,session):
    print(f"Analyzing {fn} ...")
    outfile = ( outdir / fn.stem ).with_suffix( ".fit")
    if not checkiffileneedsupdate([fn],outfile):
        print(f"No update needed for output file {outfile}")
        return
    target = parse_target(fn)
    std_field = bool(re.match(r"SA\d+(_\w+)?|GD\d+(_\w+)?|F\d+",target))
    print(f"Target: {target}",f"std_field: {std_field}")
    with fits.open(fn) as hdul:
        image = hdul[0]
        fwhmpx = image.header['FWHMPX']
        bkmean = image.header["BKMEAN"]
        bkstd = image.header["BKSTD"]
        exposure = float(image.header["EXPOSURE"])
        print(f"FWHM: {fwhmpx} pix, bkg: {bkmean}, std: {bkstd}")
        wcs = WCS(image.header)
        #bkg2d = Background2D(image.data, 64, filter_size = 3)

        daofind = DAOStarFinder(fwhm=fwhmpx, threshold = 5 * bkstd)
        sources = daofind(image.data - bkmean)
        positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
        skypositions = SkyCoord.from_pixel(sources["xcentroid"],sources["ycentroid"],wcs)
        aperture_R = fwhmpx*1.75
        annulus_Rin = fwhmpx*5
        annulus_Rout = fwhmpx*9
        annuluses = CircularAnnulus(positions, r_in=annulus_Rin, r_out=annulus_Rout)
        apertures = CircularAperture(positions, r=aperture_R)

        bkgstats = ApertureStats(image.data,annuluses,sigma_clip=SigmaClip(sigma=3.0,maxiters=10))
        aperstats = ApertureStats(image.data, apertures,sigma_clip=None)

        flux = aperstats.sum
        flux_bkg = bkgstats.mean * aperstats.sum_aper_area.value
        flux_bkg_sub = flux-flux_bkg
        instmag = -2.5 * np.log10(flux_bkg_sub/exposure)
        
        phottable = QTable([instmag,flux_bkg_sub,flux,flux_bkg,sources["xcentroid"],sources["ycentroid"],skypositions.ra.to_string("deg"),skypositions.dec.to_string("deg")],
                            names=("Instrumental Magnitude","Flux - Background", "Flux", "Background Flux","x","y","ra","dec"),
                            meta={"name": "PHOTTABLE","R":aperture_R,"RIN": annulus_Rin, "ROUT": annulus_Rout}
                        )

        vsx_table, vsp_table = combine_photometry_vsx_vsp(image,phottable,session,std_field)

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
        vsx_table_hdu = fits.BinTableHDU(vsx_table,name="VSX")
        vsp_table_hdu = fits.BinTableHDU(vsp_table,name="VSP")
        hdul_out.append(image)
        hdul_out.append(phottable_hdu)
        hdul_out.append(sources_hdu)
        hdul_out.append(aperturetable_hdu)
        hdul_out.append(annulustable_hdu)
        hdul_out.append(vsx_table_hdu)
        hdul_out.append(vsp_table_hdu)
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

    session = requests_cache.CachedSession()
    for infile in infiles:
        analyze(infile,outdir,session)
        

if __name__ == "__main__":
    main()
