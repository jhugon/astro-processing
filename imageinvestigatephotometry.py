#!/usr/bin/env python3

from pathlib import Path
import re
import io
import xml.etree.ElementTree as XMLElementTree
import requests_cache

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
    result = QTable(rows=newdata)
    return result

def load_vsx(target,session=None):
    url = f"http://www.aavso.org/vsx/index.php"
    params = {
        "view": "query.votable",
        "ident": target.replace("_"," "),
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
            for td in tabledata.iter("TD"):
                fielddata.append(td.text)
    for i in range(len(fielddata)):
        if fieldnames[i] == "Coords(J2000)":
            ra, dec = fielddata[i].split(',')
            fielddata[i] = SkyCoord(ra=ra,dec=dec,unit=u.deg)
            fieldnames[i] = "skypos"
        try:
            fielddata[i] = float(fielddata[i])
        except ValueError:
            pass
        except TypeError:
            pass
    result = Table(rows=[fielddata],names=fieldnames)
    return result

def add_phot_to_vsp_table(phot,vsp):
    idx, d2d, _ = vsp["skypos"].match_to_catalog_sky(phot["skypos"])
    result = vsp.copy()
    del result["skypos"]
    result["Match Distance"] = d2d.to(u.arcsec)
    for colname in ["Instrumental Magnitude","Flux","Background Flux","skypos","imagepos"]:
        result[colname] = phot[colname][idx]
    return result

def add_phot_to_vsx_table(phot,vsx):
    idx, d2d, _ = vsx["skypos"].match_to_catalog_sky(phot["skypos"])
    result = vsx.copy()
    del result["skypos"]
    result["Match Distance"] = d2d.to(u.arcsec)
    for colname in ["Instrumental Magnitude","Flux","Background Flux","skypos","imagepos"]:
        result[colname] = phot[colname][idx]
    return result

def analyze(fn,session):
    print(f"Analyzing {fn} ...")
    target = parse_target(fn)
    std_field = bool(re.match(r"SA\d+(_\w+)?|GD\d+(_\w+)?|F\d+",target))
    print(f"Target: {target}",f"std_field: {std_field}")
    with fits.open(fn) as hdul:
        image = hdul[0]
        wcs = WCS(image.header)

        photometry = QTable.read(hdul[1])
        photometry["skypos"] = SkyCoord(ra=photometry["ra"],dec=photometry["dec"],unit=u.deg)
        photometry["imagepos"] = np.transpose((photometry["x"],photometry["y"]))
        del photometry["x"]
        del photometry["y"]
        del photometry["ra"]
        del photometry["dec"]

        filtername = image.header["filter"]
        R = photometry.meta["R"]
        Rin = photometry.meta["RIN"]
        Rout = photometry.meta["ROUT"]

        vsp_table = load_vsp(image.header["RA"],image.header["DEC"],std_field,session,filtername=filtername)
        vsx_table = load_vsx(target,session)
        #target_pos = SkyCoord(ra=image.header["RA"],dec=image.header["DEC"],unit=(u.hourangle,u.deg))

        combined_vsp_table = add_phot_to_vsp_table(photometry,vsp_table)
        combined_vsx_table = add_phot_to_vsx_table(photometry,vsx_table)

        apertures = CircularAperture(photometry["imagepos"],r=R)
        vsp_apertures = CircularAperture(combined_vsp_table["imagepos"],r=R)
        vsx_apertures = CircularAperture(combined_vsx_table["imagepos"],r=R)

        print("VSP Stars:")
        print(combined_vsp_table)

        print("VSX Stars:")
        print(combined_vsx_table)

        norm = simple_norm(image.data,'sqrt',percent=99)
        plt.imshow(image.data,norm=norm,interpolation="nearest")
        ap_patches = apertures.plot(color='white')
        vsp_ap_patches = vsp_apertures.plot(color='purple',lw=2)
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

    session = requests_cache.CachedSession()
    for infile in args.infile:
        analyze(infile,session)
        

if __name__ == "__main__":
    main()
