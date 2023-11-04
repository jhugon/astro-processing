#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import requests_cache

from astropy import units as u
from astropy.io import fits, votable
from astropy.stats import sigma_clipped_stats, SigmaClip
import astropy.table
from astropy.table import Table, QTable
from astropy.time import Time, TimeDelta

from imagephotometry import load_vsx

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

def combine_vsp_vsx_tables(vsp: Table,vsx: Table,filtername: str) -> QTable:
    result = vsp.copy()
    names = ("auid","measmag","catmag","catmagerr","matchdist","rawpeak","isvsp")
    auids = []
    measmags = []
    catmags = []
    catmagerrs = []
    matchdists = []
    rawpeaks = []
    isvsps = []
    for row in vsp:
        auids.append(row["auid"])
        measmags.append(row["Instrumental Magnitude"]*u.mag)
        catmags.append(row[filtername])
        catmagerrs.append(row[filtername+"error"])
        matchdists.append(row["Match Distance"])
        rawpeaks.append(row["Raw Peak"]*u.adu)
        isvsps.append(True)
    if vsx:
        vsx_good_auids = vsx[np.logical_not(vsx["AUID"].mask)]
        for row in vsx_good_auids:
            auids.append(row["AUID"])
            measmags.append(row["Instrumental Magnitude"]*u.mag)
            catmags.append(float('nan')*u.mag)
            catmagerrs.append(float('nan')*u.mag)
            matchdists.append(row["Match Distance"])
            rawpeaks.append(row["Raw Peak"]*u.adu)
            isvsps.append(False)
    meta = {"std_field":vsp.meta["std_field"],"filter":filtername,"target":vsp.meta["TARGET"]}
    result = QTable((auids,measmags,catmags,catmagerrs,matchdists,rawpeaks,isvsps),names=names,meta=meta)
    return result

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
    if filtername == "I":
        return QTable()
    catMag = {}
    catMagErr = {}
    matchDistance = {}
    measMag = {}
    rawpeak = {}
    isVSP = {}
    jds = []
    targets = set()
    for fn in fns:
        vsp, vsx = get_vsp_vsx_tables(fn)
        table = combine_vsp_vsx_tables(vsp,vsx,filtername)
        targets.add(table.meta["target"])
        jds.append(get_image_jd_filter(fn)[0])
        for row in table:
            auid = row["auid"]
            catMag[auid] = row["catmag"]
            catMagErr[auid] = row["catmagerr"]
            isVSP[auid] = row["isvsp"]
            try:
                matchDistance[auid] = max(row["matchdist"],matchDistance[auid])
            except KeyError:
                matchDistance[auid] = row["matchdist"]
                measMag[auid] = [row["measmag"]]
                rawpeak[auid] = [row["rawpeak"]]
            else:
                measMag[auid].append(row["measmag"])
                rawpeak[auid].append(row["rawpeak"])
    assert(len(targets)==1)
    avgMag = {key:u.Quantity(measMag[key]).mean() for key in measMag}
    stdMag = {key:u.Quantity(measMag[key]).std()/np.sqrt(len(measMag[key])) for key in measMag}
    avgrawpeak = {key:u.Quantity(rawpeak[key]).mean() for key in rawpeak}
    auids = sorted(catMag.keys())
    result = QTable(
        [
            auids,
            [avgMag[auid] for auid in auids],
            [stdMag[auid] for auid in auids],
            [catMag[auid] for auid in auids],
            [catMagErr[auid] for auid in auids],
            [matchDistance[auid] for auid in auids],
            [avgrawpeak[auid] for auid in auids],
            [isVSP[auid] for auid in auids],
        ],
        names = ("auid","measmag","measmagerr","catmag","catmagerr","matchdist","rawpeak","isvsp"),
        meta = {"filter":filtername,"jds":jds,"target":targets.pop()}
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
        if len(tables[iObs]) == 0:
            iObs += 1
            continue
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
    targets = set()
    for auid in auids:
        newrow = {"auid": auid,"matchdist":-20.*u.arcsec}
        for table in tables:
            filtername = table.meta["filter"]
            targets.add(table.meta["target"])
            row = table.loc[auid]
            newrow["matchdist"] = max(newrow["matchdist"],row["matchdist"])
            newrow[filtername+"meas"] = row["measmag"]
            newrow[filtername+"measerr"] = row["measmagerr"]
            newrow[filtername+"cat"] = row["catmag"]
            newrow[filtername+"caterr"] = row["catmagerr"]
            newrow[filtername+"rawpeak"] = row["rawpeak"]
            newrow["isvsp"] = row["isvsp"]
        newrows.append(newrow)
    assert(len(targets)==1)
    result = QTable(rows=newrows,meta={"jds":newjds,"target":targets.pop()})
    return result

def load_group_by_runs_filters(fns: [Path]) -> [[QTable]]:
    runs = seperate_runs(fns)
    result = []
    for run in runs:
        filter_blocks = blocks_filters(run)
        filter_block_vsp_averages = [average_vsp_tables(filter_block,filtername) for filtername,filter_block in filter_blocks]
        filter_group_lists = group_filters(filter_block_vsp_averages)
        # Below has a measurement and catalog value for each observed filter, but there are still multiple observations
        filters_grouped_list = [combine_filters(x) for x in filter_group_lists]
        result.append(filters_grouped_list)
    return result

def calibrate(tables: [[QTable]]) -> None:
    markersize = 2 ** 2 # default is 6 ** 2

    selectorforoffset = lambda table: (table["matchdist"] < 10*u.arcsec) & table["isvsp"]
    selectorforplots = selectorforoffset
    #selectorforplots = lambda table: selectorforoffset(table) & (table["Vcat"] > 11.5*u.mag) & (table["Vcat"] < 14*u.mag) & (table["Bcat"] < 14*u.mag)

    newtables = []
    for run in tables:
        for obs in run:
            obs["target"] = obs.meta["target"]
            jds = Time(obs.meta["jds"]) # list of Times to Time with list
            obs["jd"] = jds.mean()
            obsforoffset = obs[selectorforoffset(obs)]
            for filtername in ["B","V"]:
                offset = (obsforoffset[filtername+"meas"]-obsforoffset[filtername+"cat"]).mean()
                obs[filtername+"calib"] = obs[filtername+"meas"]-offset
            newtables.append(obs)

    alltable = astropy.table.vstack(newtables,metadata_conflicts="silent")
    alltable = alltable[selectorforplots(alltable)]

    for filtername in ["B","V"]:
        calibdiffs = alltable[filtername+"calib"]-alltable[filtername+"cat"]
        fig, (ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(6,figsize=(8,10),constrained_layout=True)
        yaxis_label = f"{filtername} Cal - Cat [mag]"
        fig.suptitle("Offset Calibration Only--No Color Calibration")
        ax1.scatter(alltable["Bmeas"]-alltable["Vmeas"],calibdiffs,s=markersize)
        ax1.set_xlabel("Instrumental B-V [mag]")
        ax1.set_ylabel(yaxis_label)
        ax1.grid(True)
        ax2.scatter(alltable["Bcat"]-alltable["Vcat"],calibdiffs,s=markersize)
        ax2.set_xlabel("Catalog B-V [mag]")
        ax2.set_ylabel(yaxis_label)
        ax2.grid(True)
        ax3.scatter(alltable[filtername+"cat"],calibdiffs,s=markersize)
        ax3.set_xlabel(f"Catalog {filtername} [mag]")
        ax3.set_ylabel(yaxis_label)
        ax3.grid(True)
        ax4.scatter(alltable[filtername+"measerr"],calibdiffs,s=markersize)
        ax4.set_xlabel(f"Estimated Measured {filtername} Uncertainty [mag]")
        ax4.set_ylabel(yaxis_label)
        ax4.set_xscale("log")
        ax4.grid(True)
        ax5.scatter(alltable["matchdist"],calibdiffs,s=markersize)
        ax5.set_xlabel("Match Distance [arcsec]")
        ax5.set_ylabel(yaxis_label)
        ax5.set_xscale("log")
        ax5.grid(True)
        ax6.scatter(alltable[filtername+"rawpeak"],calibdiffs,s=markersize)
        ax6.set_xlabel("Peak Value [ADU]")
        ax6.set_ylabel(yaxis_label)
        ax6.grid(True)
        fig.savefig(f"CalibOffsetOnly_{filtername}.png")
        
        fig, (ax1,ax2) = plt.subplots(2,figsize=(8,10),constrained_layout=True)
        fig.suptitle("Offset Calibration Only--No Color Calibration")
        ax1.scatter(alltable[filtername+"rawpeak"],calibdiffs,s=markersize)
        ax1.set_xlabel("Peak Value [ADU]")
        ax1.set_ylabel(yaxis_label)
        ax1.grid(True)
        ax2.scatter(alltable[filtername+"rawpeak"],calibdiffs,s=markersize)
        ax2.set_xlabel("Peak Value [ADU]")
        ax2.set_ylabel(yaxis_label)
        ax2.grid(True)
        ax1.set_xlim(None,10000)
        ax2.set_xlim(60000,None)
        fig.savefig(f"CalibOffsetOnlyMeasErrVRawPeak_{filtername}.png")
        

def lightcurve(tables: [[QTable]], targetname: str) -> None:
    session = requests_cache.CachedSession()
    vsx = load_vsx(ident=targetname,session=session)
    targetname = vsx[0]["Name"]
    targetauid = vsx[0]["AUID"]
    targetepoch = vsx[0]["Epoch"]
    targetperiod = vsx[0]["Period"]
    targetrisepercent = vsx[0]["RiseDuration"]
    targetriseduration = None
    if targetepoch:
        targetepoch = Time(2400000.+targetepoch,format="jd",scale="tt")
    if targetperiod:
        targetperiod = TimeDelta(targetperiod*u.day)
    if targetrisepercent:
        targetriseduration = targetrisepercent/100. * targetperiod
        targetrisepercent = targetrisepercent * u.percent
    print(f"Target: {targetname} AUID: {targetauid} Epoch: {targetepoch} Period: {targetperiod} Rise Percent: {targetrisepercent} Rise Duration: {targetriseduration}")
    compauids = set()
    for run in tables:
        for obs in run:
            goodcomps = obs[(obs["matchdist"] < 10*u.arcsec) & obs["isvsp"]]
            for auid in goodcomps["auid"]:
                compauids.add(str(auid))
    compauids = sorted(compauids)
    jdsV = [[] for _ in compauids]
    phaseV = [[] for _ in compauids]
    magsV = [[] for _ in compauids]
    magerrsV = [[] for _ in compauids]
    jdsB = [[] for _ in compauids]
    phaseB = [[] for _ in compauids]
    magsB = [[] for _ in compauids]
    magerrsB = [[] for _ in compauids]
    for run in tables:
        for obs in run:
            measrow = obs[obs["auid"] == targetauid]
            if measrow["matchdist"] > 10. * u.arcsec:
                continue
            jd = Time(obs.meta["jds"]).mean()
            for iComp, compauid in enumerate(compauids):
                comprow = obs[obs["auid"] == compauid]
                if comprow["matchdist"] > 10. * u.arcsec:
                    continue
                magV = measrow["Vmeas"] - comprow["Vmeas"] + comprow["Vcat"]
                magB = measrow["Bmeas"] - comprow["Bmeas"] + comprow["Bcat"]
                magsV[iComp].append(magV.value[0])
                magsB[iComp].append(magB.value[0])
                magerrsV[iComp].append(measrow["Vmeaserr"].value[0])
                magerrsB[iComp].append(measrow["Bmeaserr"].value[0])
                jdsV[iComp].append(jd.value)
                jdsB[iComp].append(jd.value)
                phase = (jd-targetepoch).value % targetperiod.value
                phaseV[iComp].append(phase)
                phaseB[iComp].append(phase)
    fig, ((ax1,ax3),(ax2,ax4)) = plt.subplots(2,2,figsize=(6,6),constrained_layout=True)#,sharex=True,sharey=True)
    fig.suptitle(f"{targetname}, No Color Calibration")
    for iComp, comp in enumerate(compauids):
        ax1.errorbar(jdsB[iComp],magsB[iComp],yerr=magerrsB[iComp],ls="")
        ax2.errorbar(jdsV[iComp],magsV[iComp],yerr=magerrsV[iComp],ls="")
        ax3.errorbar(phaseB[iComp],magsB[iComp],yerr=magerrsB[iComp],ls="")
        ax4.errorbar(phaseV[iComp],magsV[iComp],yerr=magerrsV[iComp],ls="")
    ax1.set_ylabel(f"B [mag]")
    ax2.set_xlabel("JD")
    ax2.set_ylabel(f"V [mag]")
    ax4.set_xlabel(f"Phase [T = {targetperiod} d]")
    ax3.set_xlim(0.,1.)
    ax4.set_xlim(0.,1.)
    fig.savefig(f"Phot-{targetname.replace(' ','_')}.png")

def main():

    import argparse
    parser = argparse.ArgumentParser(
        prog="photometryanalysis.py",
        description="If --target is present, then computes the light curve for the given target star. Otherwise, produces calibration plots assuming standard fields"
    )
    parser.add_argument("infile",type=Path,nargs="+",help="Input fits file(s)")
    parser.add_argument("--target",type=str,help="Target star AUID")
    parser.add_argument("--comp",type=str,help="Comparison star AUID")
    parser.add_argument("--check",type=str,help="Check star AUID")

    args = parser.parse_args()

    tables = load_group_by_runs_filters(args.infile)
    if args.target:
        lightcurve(tables,args.target)
    else:
        calibrate(tables)

if __name__ == "__main__":
    main()
