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

def get_image_jd_filter(fn: Path) -> (Time,str):
    with fits.open(fn) as hdul:
        image = hdul[0]
        dateobsstr = image.header["date-obs"]
        dateobs = Time(dateobsstr,format='isot',scale='utc')
        jd = dateobs.tt
        jd.format = "jd"
        filtername = image.header["filter"]
        return jd, filtername

def get_vsp_vsx_tables(fn):
    with fits.open(fn) as hdul:
        vsp_table = None
        vsx_table = None
        for hdu in hdul:
            if hdu.name == "VSP":
                vsp_table = QTable.read(hdu)
            elif hdu.name == "VSX":
                try:
                    vsx_table = QTable.read(hdu)
                    vsx_table["x"]
                except KeyError:
                    vsx_table = None
        return vsp_table,vsx_table

def seperate_runs(fns: [Path]) -> [[Path]]:
    """
    Breaks up file list into groups of files "runs" seperated by 6000 seconds
    or more. All files seperated by 6000 s or less are grouped together
    """
    if len(fns) < 2:
        return [fns]
    fnjds = []
    for fn in fns:
        jd, _ = get_image_jd_filter(fn)
        fnjds.append((fn,jd))
    fnjds = sorted(fnjds,key= lambda x: x[1])
    iresults = [[0]]
    for iRow in range(1,len(fnjds)):
        dt = fnjds[iRow][1]-fnjds[iRow-1][1]
        if dt > TimeDelta(6000*u.second):
            iresults.append([iRow])
        else:
            iresults[-1].append(iRow)
    result = []
    for run in iresults:
        result.append([])
        for ir in run:
            result[-1].append(fnjds[ir][0])
    #for iRun, run in enumerate(result):
    #    print("Run: ",iRun)
    #    for fn in run:
    #        print(fn)
    #breakpoint()
    return result

def blocks_filters(fns: [Path]) -> [(str,[Path])]:
    """
    Groups filenames together that are a contigous group of the same filter (for e.g. averaging later)
    """
    fns = sorted(fns, key=lambda x: get_image_jd_filter(x)[0])
    blocks = [(get_image_jd_filter(fns[0])[1],[0])]
    for i in range(1,len(fns)):
        current_filter = get_image_jd_filter(fns[i])[1]
        last_filter = blocks[-1][0]
        if current_filter == last_filter:
            blocks[-1][1].append(i)
        else:
            blocks.append((current_filter,[i]))
    result = []
    for filtername,block in blocks:
        result.append((filtername,[fns[i] for i in block]))
    #for filtername,block in result:
    #    print("Filter Block: ",filtername)
    #    for fn in block:
    #        print(fn)
    #breakpoint()
    return result

def average_vsp_tables(fns,filtername):
    catMag = {}
    catMagErr = {}
    matchDistance = {}
    measMag = {}
    jds = []
    for fn in fns:
        vsp, _ = get_vsp_vsx_tables(fn)
        jds.append(get_image_jd_filter(fn)[0])
        for row in vsp:
            auid = row["auid"]
            catMag[auid] = row[filtername]
            catMagErr[auid] = row[filtername+"error"]
            try:
                matchDistance[auid] = max(row["Match Distance"],matchDistance[auid])
            except KeyError:
                matchDistance[auid] = row["Match Distance"]
                measMag[auid] = [row["Instrumental Magnitude"]]
            else:
                measMag[auid].append(row["Instrumental Magnitude"])
    avgMag = {key:np.mean(measMag[key]) for key in measMag}
    stdMag = {key:np.std(measMag[key]) for key in measMag}
    auids = sorted(catMag.keys())
    result = QTable(
        [
            auids,
            [avgMag[auid] for auid in auids],
            [stdMag[auid] for auid in auids],
            [catMag[auid] for auid in auids],
            [catMagErr[auid] for auid in auids],
            [matchDistance[auid] for auid in auids],
        ],
        names = ("auid","measmag","measmagerr","catmag","catmagerr","matchdist"),
        meta = {"filter":filtername,"jds":jds}
    )
    return result

def group_filters(tables: [Table]) -> [[Table]]:
    """
    Groups tables of each filter together, so you have [[B,V],[B,V],...]
    """
    iObs = 0
    nObs = len(tables)
    filter1 = "B"
    filter2 = "V"
    result_list = []
    while iObs < nObs:
        ifilter = tables[iObs].meta["filter"]
        if ifilter == filter1:
            jObs = iObs + 1
            foundfilter2 = False
            while jObs < nObs:
                jfilter = tables[jObs].meta["filter"]
                if jfilter == filter2:
                    result_list.append([tables[iObs],tables[jObs]])
                    iObs = jObs + 1
                    foundfilter2 = True
                    break
                else:
                    jObs += 1
            if not foundfilter2:
                iObs += 1
        else:
            iObs += 1
    #for table in tables:
    #    print(table.meta["filter"])
    #print()
    #for group in result_list:
    #    print([x.meta["filter"] for x in group])
    #breakpoint()
    return result_list
    
def combine_filters(tables: [Table]) -> Table:
    auids = set()
    for table in tables:
        auids |= set(table["auid"])
    auids = sorted(auids)
    newjds = []
    for table in tables:
        table.add_index("auid")
        newjds += table.meta["jds"]
    newrows = []
    for auid in auids:
        newrow = {"auid": auid,"matchdist":-20.*u.arcsec}
        for table in tables:
            filtername = table.meta["filter"]
            row = table.loc[auid]
            newrow["matchdist"] = max(newrow["matchdist"],row["matchdist"])
            newrow[filtername+"meas"] = row["measmag"]
            newrow[filtername+"measerr"] = row["measmagerr"]
            newrow[filtername+"cat"] = row["catmag"]
            newrow[filtername+"caterr"] = row["catmagerr"]
        newrows.append(newrow)
    result = QTable(rows=newrows,meta={"jds":newjds})
    breakpoint()
    return result

def combine_obs_in_run(tables: [Table]) -> Table:
    table_list = []
    for table in tables:
        jds = table.meta["jds"]


def analyze_by_file(fns: [Path]) -> None:
    runs = seperate_runs(fns)
    for run in runs:
        filter_blocks = blocks_filters(run)
        filter_block_vsp_averages = [average_vsp_tables(filter_block,filtername) for filtername,filter_block in filter_blocks]
        filter_group_lists = group_filters(filter_block_vsp_averages)
        # Below has a measurement and catalog value for each observed filter, but there are still multiple observations
        filters_grouped_list = [combine_filters(x) for x in filter_group_lists]
        all_obs_in_run = astropy.table.vstack([ for x in filters_grouped_list])

def main():

    import argparse
    parser = argparse.ArgumentParser(
        prog="imageimvestigatephotometry.py",
        description="Investiage photometry script output."
    )
    parser.add_argument("infile",type=Path,nargs="+",help="Input fits file(s)")

    args = parser.parse_args()

    analyze_by_file(args.infile)
    analyze_by_individual_star(args.infile)
        

if __name__ == "__main__":
    main()
