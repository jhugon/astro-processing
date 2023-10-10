#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.io import fits, votable
from astropy.stats import sigma_clipped_stats, SigmaClip
import astropy.table
from astropy.table import Table, QTable
from astropy.time import Time, TimeDelta

from imageanalysisplatesolve import findfilesindir, checkiffileneedsupdate

def seperate_runs(table):
    iresults = [[0]]
    for iRow in range(1,len(table)):
        dt = table[iRow]["jd"]-table[iRow-1]["jd"]
        if dt > TimeDelta(6000*u.second):
            iresults.append([iRow])
        else:
            iresults[-1].append(iRow)
    result = []
    for ir in iresults:
        result.append(table[ir])
    return result

def sum_blocks_filters(table):
    row_groups = [[0]]
    for i in range(1,len(table)):
        row = table[i]
        last_row = table[i-1]
        if row["filter"] == last_row["filter"]:
            row_groups[-1].append(i)
        else:
            row_groups.append([i])
    new_rows = []
    for row_group in row_groups:
        new_row = {"mag": table[row_group]["mag"].mean(),"jd": table[row_group]["jd"].mean(),"filter": table[row_group]["filter"][0]}
        new_rows.append(new_row)
    result = QTable(rows=new_rows,meta=table.meta)
    return result

def make_colors(table):
    #filterorder = ["U","B","V","Rc","R","I","Ic"]
    #allfilters = set(table["filter"])
    #sortedfilters = sorted(allfilters,key=filterorder.index)
    summed_blocks_table = sum_blocks_filters(table)
    iObs = 0
    nObs = len(summed_blocks_table)
    filter1 = "B"
    filter2 = "V"
    result_list = []
    while iObs < nObs:
        rowi = summed_blocks_table[iObs]
        if rowi["filter"] == filter1:
            jObs = iObs + 1
            foundfilter2 = False
            while jObs < nObs:
                rowj = summed_blocks_table[jObs]
                if rowj["filter"] == filter2:
                    result_row = {
                        filter1+"mag":rowi["mag"],
                        filter2+"mag":rowj["mag"],
                        filter1+"jd":rowi["jd"],
                        filter2+"jd":rowj["jd"],
                        "jd": rowi["jd"]+0.5*abs(rowj["jd"]-rowi["jd"]),
                        filter1+"-"+filter2: rowi["mag"]-rowj["mag"],
                    }
                    result_list.append(result_row)
                    iObs = jObs + 1
                    foundfilter2 = True
                    break
                else:
                    jObs += 1
            if not foundfilter2:
                iObs += 1
        else:
            iObs += 1
    result = QTable(rows=result_list,meta=table.meta)
    return result

def analyze(fns: [Path]) -> None:
    referenceMag = {}
    measurement = {}
    measurementtables = {}
    measurementallfilters = {}
    measurementallfilterstables = {}
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

            #print("VSP Stars:")
            #print(vsp_table)

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
                    measurementallfilters[auid].append(meas | {"filter":filtername})
                except KeyError:
                    try:
                        measurementallfilters[auid] = [meas | {"filter":filtername}]
                        measurementallfilterstables[auid] = []
                        measurement[auid][filtername] = [meas]
                        measurementtables[auid][filtername] = []
                    except KeyError:
                        measurement[auid] = {filtername: [meas]}
                        measurementtables[auid] = {filtername: []}
    for auid in measurement:
        auiddata = measurement[auid]
        measurementallfilterstables[auid] = QTable(measurementallfilters[auid],meta={"auid":auid})
        for filtername in auiddata:
            filterdata = auiddata[filtername]
            #print(auid,filtername,filterdata)
            table = QTable(filterdata,meta={"auid":auid,"filter":filtername})
            #print(auid,filtername)
            #print(table)
            #print(table["jd"] - table["jd"][0])
            measurementtables[auid][filtername] = table
    allcolortables = []
    for auid in measurementallfilterstables:
        table = measurementallfilterstables[auid]
        runs = seperate_runs(table)
        colortables = []
        for run in runs:
            colortable = make_colors(run)
            colortables.append(colortable)
        colortable = astropy.table.vstack(colortables)
        colortable["auid"] = [auid]*len(colortable)
        allcolortables.append(colortable)
    allcolortable = astropy.table.vstack(allcolortables)

    fig, ax = plt.subplots()
    refB = [referenceMag[auid]["B"] for auid in allcolortable["auid"]]
    refV = [referenceMag[auid]["V"] for auid in allcolortable["auid"]]
    refBmV = [B.value-V.value for B,V in zip(refB,refV)]
    ax.scatter(refBmV,allcolortable["B-V"].value)
    ax.set_xlabel("Catalog B-V [mag]")
    ax.set_ylabel("Measured B-V [mag]")
    fig.savefig("BminusV.png")

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
