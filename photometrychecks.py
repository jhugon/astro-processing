#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.io import fits, votable
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.table import Table, QTable
from astropy.time import Time

from imageanalysisplatesolve import findfilesindir, checkiffileneedsupdate

def analyze(fns: [Path]) -> None:
    referenceMag = {}
    measurement = {}
    measurementtables = {}
    for fn in fns:
        print(f"Analyzing {fn} ...")
        with fits.open(fn) as hdul:
            image = hdul[0]
            filtername = image.header["filter"]
            dateobsstr = image.header["date-obs"]
            dateobs = Time(dateobsstr,format='isot',scale='utc')
            jd = dateobs.tt
            jd.format = "jd"
            print(f"{dateobs} JD: {jd}")

            vsp_table = None
            vsx_table = None
            for hdu in hdul:
                if hdu.name == "VSP":
                    vsp_table = QTable.read(hdu)
                    del vsp_table["x"]
                    del vsp_table["y"]
                    del vsp_table["ra"]
                    del vsp_table["dec"]
                elif hdu.name == "VSX":
                    try:
                        vsx_table = QTable.read(hdu)
                        del vsx_table["x"]
                        del vsx_table["y"]
                        del vsx_table["ra"]
                        del vsx_table["dec"]
                    except KeyError:
                        vsx_table = None

            std_field = vsp_table.meta["std_field"]

            if std_field:
                print("This is a standard field")

            print("VSP Stars:")
            print(vsp_table)

            for row in vsp_table:
                #breakpoint()
                auid = row["auid"]
                refmags = {
                    filtername : row[filtername],
                    filtername + "error" : row[filtername+"error"],
                }
                try:
                    referenceMag[auid] |= refmags
                except KeyError:
                    referenceMag[auid] = refmags

                meas = {
                    "mag": row["Instrumental Magnitude"]*u.mag,
                    "jd": jd,
                    "s/b": row["Flux"]/row["Background Flux"],
                }
                try:
                    measurement[auid][filtername].append(meas)
                except KeyError:
                    try:
                        measurement[auid][filtername] = [meas]
                        measurementtables[auid][filtername] = []
                    except KeyError:
                        measurement[auid] = {filtername: [meas]}
                        measurementtables[auid] = {filtername: []}
    for auid in measurement:
        auiddata = measurement[auid]
        for filtername in auiddata:
            filterdata = auiddata[filtername]
            #print(auid,filtername,filterdata)
            table = QTable(filterdata,meta={"auid":auid,"filter":filtername})
            print(auid,filtername)
            print(table)
            measurementtables[auid][filtername] = table



def main():

    import argparse
    parser = argparse.ArgumentParser(
        prog="imageimvestigatephotometry.py",
        description="Investiage photometry script output."
    )
    parser.add_argument("infile",type=Path,nargs="+",help="Input fits file(s)")

    args = parser.parse_args()

    analyze(args.infile)
        

if __name__ == "__main__":
    main()
