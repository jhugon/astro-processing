#!/usr/bin/env python3

from pathlib import Path
import sys

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
from photometryanalysis import get_vsp_vsx_tables, combine_vsp_vsx_tables, get_image_jd_filter

def load_files_by_filter(fns: [Path]) -> {str:[QTable]}:
    tables = {}
    for fn in fns:
        jd, filtername = get_image_jd_filter(fn)
        vsp, vsx = get_vsp_vsx_tables(fn)
        table = combine_vsp_vsx_tables(vsp,vsx,filtername)
        table.meta["jd"] = jd
        try:
            tables[filtername].append(table)
        except KeyError:
            tables[filtername] = [table]
    result = {}
    for filtername in tables:
        result[filtername] = sorted(tables[filtername],key=lambda x: x.meta["jd"])
    return result

def analyze_variable_star(tablesbyfilter: {str:[QTable]},targetname: str,compauid = None,checkauid = None) -> None:
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
    checkauids = set()
    fig, ((ax1,ax3),(ax2,ax4)) = plt.subplots(2,2,figsize=(6,6),constrained_layout=True)#,sharex=True,sharey=True)
    fig.suptitle(f"{targetname}, No Color Calibration")
    for filtername in tablesbyfilter:
        tables = tablesbyfilter[filtername]
        maglist = []
        jdlist = []
        phaselist = []
        for table in tables:
            jd = table.meta["jd"]
            table.add_index("auid")
            target = table.loc[targetauid]
            if not compauid:
                print(f"Choose a comp star for {targetname}, and add it to the command line with flag --comp")
                print(table[table["isvsp"]])
                sys.exit(1)
            comp = table.loc[compauid]
            if target["matchdist"] > 10*u.arcsec or comp["matchdist"] > 10*u.arcsec:
                continue
            mag = target["measmag"] - comp["measmag"] + comp["catmag"]
            maglist.append(mag)
            jdlist.append(jd)
            phase = (jd-targetepoch).value % targetperiod.value
            phaselist.append(phase)
            for row in table:
                auid = row["auid"]
                if row["matchdist"] > 10*u.arcsec or auid == targetauid or auid == compauid or (not row["isvsp"]):
                    continue
                checkauids.add(auid)
        mags = u.Quantity(maglist)
        jds = Time(jdlist)
        if filtername == "B":
            ax1.scatter(jds.value,mags.value)
            ax3.scatter(phaselist,mags.value)
        elif filtername == "V":
            ax2.scatter(jds.value,mags.value)
            ax4.scatter(phaselist,mags.value)
    ax1.set_ylabel(f"B [mag]")
    ax2.set_xlabel("JD")
    ax2.set_ylabel(f"V [mag]")
    ax4.set_xlabel(f"Phase [T = {targetperiod} d]")
    ax3.set_xlim(0.,1.)
    ax4.set_xlim(0.,1.)
    fig.savefig(f"PhotNoColor-{targetname.replace(' ','_')}.png")

    ## Plot things on all check stars
    checkauids = sorted(checkauids)
    figcheckvtime, (axcheckvtimeB,axcheckvtimeV) = plt.subplots(2,figsize=(8,8),constrained_layout=True,sharex=True)
    figcheckvtime.suptitle(f"{targetname} Check Stars, No Color Calibration")
    for filtername in tablesbyfilter:
        tables = tablesbyfilter[filtername]
        checkmagvjd = {auid: [] for auid in checkauids}
        checkmagerrvjd = {auid: [] for auid in checkauids}
        compcatmag = float("nan")
        for table in tables:
            jd = table.meta["jd"]
            table.add_index("auid")
            for iCheck, auid in enumerate(checkauids):
                check = table.loc[auid]
                comp = table.loc[compauid]
                if check["matchdist"] > 10*u.arcsec or comp["matchdist"] > 10*u.arcsec:
                    continue
                mag = check["measmag"] - comp["measmag"] + comp["catmag"]
                magerr = mag-check["catmag"]
                checkmagvjd[auid].append((jd.value,mag.value))
                checkmagerrvjd[auid].append((jd.value,magerr.value))
                compcatmag = comp["catmag"]
        print(f"Check stars for filter: {filtername}, comp: {compauid}, comp cat: {compcatmag:5.2f}")
        for key in checkmagvjd:
            checkmagvjd[key] = np.array(checkmagvjd[key])
            checkmagerrvjd[key] = np.array(checkmagerrvjd[key])
        for iCheck, auid in enumerate(checkauids):
            print(f"AUID: {auid} mean(meas): {np.mean(checkmagvjd[auid][:,1]):5.2f} mag, mean(meas - cat): {np.mean(checkmagerrvjd[auid][:,1]):6.3f} mag, std(meas - cat): {np.std(checkmagerrvjd[auid][:,1]):5.3f} mag")
            if filtername == "B":
                axcheckvtimeB.scatter(checkmagvjd[auid][:,0],checkmagerrvjd[auid][:,1],label=auid)
            elif filtername == "V":
                axcheckvtimeV.scatter(checkmagvjd[auid][:,0],checkmagerrvjd[auid][:,1],label=auid)
    axcheckvtimeB.set_ylabel(r"B$_\mathrm{meas}$-B$_\mathrm{cat}$ [mag]")
    axcheckvtimeV.set_ylabel(r"V$_\mathrm{meas}$-V$_\mathrm{cat}$ [mag]")
    axcheckvtimeV.set_xlabel("JD")
    axcheckvtimeB.legend()
    figcheckvtime.savefig(f"PhotNoColorCheckStarsVTime-{targetname.replace(' ','_')}.png")
                
        

def analyze_vsp_stars(tablesbyfilter: {str:[QTable]}) -> None:
    for filtername in tablesbyfilter:
        tables = tablesbyfilter[filtername]
        measmaglist = []
        catmaglist = []
        matchdistancelist = []
        measerrlist = []
        for table in tables:
            obs = table[(table["matchdist"] < 10*u.arcsec) & table["isvsp"]]
            breakpoint()
            jd = obs.meta["jd"]
            diff = obs["measmag"]-obs["catmag"]
            meas = obs["measmag"]-diff.mean()



def main():

    import argparse
    parser = argparse.ArgumentParser(
        prog="photometryanalysisnocolor.py",
        description="If --target is present, then computes the light curve for the given target star. Otherwise, produces calibration plots assuming standard fields"
    )
    parser.add_argument("infile",type=Path,nargs="+",help="Input fits file(s)")
    parser.add_argument("--target",type=str,help="Target star AUID")
    parser.add_argument("--comp",type=str,help="Comparison star AUID")
    parser.add_argument("--check",type=str,help="Check star AUID")

    args = parser.parse_args()

    tables = load_files_by_filter(args.infile)
    if args.target:
        analyze_variable_star(tables,args.target,args.comp,args.check)
    else:
        analyze_vsp_stars(tables)

if __name__ == "__main__":
    main()
