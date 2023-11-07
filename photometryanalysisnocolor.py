#!/usr/bin/env python3

from dataclasses import dataclass, asdict
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
from photometryanalysis import get_vsp_vsx_tables, combine_vsp_vsx_tables, get_image_jd_filter, get_altaz_siderealtime, get_header_fwhmpx

def load_files_by_filter(fns: [Path]) -> {str:[QTable]}:
    tables = {}
    for fn in fns:
        jd, filtername = get_image_jd_filter(fn)
        altaz, sidereal = get_altaz_siderealtime(fn)
        fwhmpx = get_header_fwhmpx(fn)
        vsp, vsx = get_vsp_vsx_tables(fn)
        table = combine_vsp_vsx_tables(vsp,vsx,filtername)
        table.meta["jd"] = jd
        table.meta["altaz"] = altaz
        table.meta["sidereal"] = sidereal
        table.meta["fwhmpx"] = fwhmpx
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
                

@dataclass
class ObsSummaryData:
    N: int
    target: str
    filtername: str
    jd: Time
    sidereal: Time
    alt: float
    ssr: u.mag**2
    zeropoint: u.mag
    fwhmpx: u.pixel

def analyze_vsp_stars(tablesbyfilter: {str:[QTable]}) -> None:

    markersize = 4**2 # default is 6**2
    rawpeakmin = 3200*u.adu
    rawpeakmax = 64000*u.adu
    imageedgewidth = 20*u.pixel
    selector = lambda table: (table["matchdist"] < 10*u.arcsec) & table["isvsp"] \
                                        & (table["rawpeak"] > rawpeakmin) & (table["rawpeak"] < rawpeakmax) \
                                        & (table["x"] > imageedgewidth) & (table["x"] < (table.meta["NPIXX"]*u.pixel - imageedgewidth)) \
                                        & (table["y"] > imageedgewidth) & (table["y"] < (table.meta["NPIXY"]*u.pixel - imageedgewidth)) \

    for filtername in tablesbyfilter:
        obslist = []
        for table in tablesbyfilter[filtername]:
            obs = table[selector(table)]
            if len(obs) == 0:
                continue
            obs["jd"] = obs.meta["jd"]
            instdiff = obs["measmag"]-obs["catmag"]
            zeropoint = instdiff.mean()
            obs["calibmag"] = obs["measmag"] - zeropoint
            obslist.append(obs)
        alltable = astropy.table.vstack(obslist,metadata_conflicts="silent")

        print(alltable.colnames)

        calibdiffs = alltable["calibmag"]-alltable["catmag"]
        fig, (ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(6,figsize=(8,10),constrained_layout=True)
        yaxis_label = f"{filtername} Cal - Cat [mag]"
        fig.suptitle("Zeropoint Calibration Only--No Color Calibration")
        ax1.scatter(alltable["x"],calibdiffs,s=markersize)
        ax1.set_xlabel(f"Image x position [pixel]")
        ax1.set_ylabel(yaxis_label)
        ax1.grid(True)
        ax2.scatter(alltable["y"],calibdiffs,s=markersize)
        ax2.set_xlabel(f"Image y position [pixel]")
        ax2.set_ylabel(yaxis_label)
        ax2.grid(True)
        ax3.scatter(alltable["catmag"],calibdiffs,s=markersize)
        ax3.set_xlabel(f"Catalog {filtername} [mag]")
        ax3.set_ylabel(yaxis_label)
        ax3.grid(True)
        ax5.scatter(alltable["matchdist"],calibdiffs,s=markersize)
        ax5.set_xlabel("Match Distance [arcsec]")
        ax5.set_ylabel(yaxis_label)
        ax5.set_xscale("log")
        ax5.grid(True)
        ax6.scatter(alltable["rawpeak"],calibdiffs,s=markersize)
        ax6.set_xlabel("Raw Peak Value [ADU]")
        ax6.set_ylabel(yaxis_label)
        ax6.grid(True)
        fig.savefig(f"CalibNoColor_{filtername}.png")
        
    #### Observation summaries ####

    obssummaries = []
    for filtername in tablesbyfilter:
        for table in tablesbyfilter[filtername]:
            obs = table[selector(table)]
            N = len(obs)
            target = obs.meta["TARGET"]
            jd = obs.meta["jd"]
            sidereal = obs.meta["sidereal"]
            alt = obs.meta["altaz"].alt.value
            instdiff = obs["measmag"]-obs["catmag"]
            zeropoint = instdiff.mean()
            residuals = obs["measmag"] - zeropoint - obs["catmag"]
            ssr = (residuals**2).sum()
            osd = ObsSummaryData(N,target,filtername,jd,sidereal,alt,ssr,zeropoint,obs.meta["fwhmpx"])
            obssummaries.append(osd)
    obssummarytable = QTable([asdict(x) for x in obssummaries])
    obssummarytableV = obssummarytable[obssummarytable["filtername"]=="V"]
    obssummarytableB = obssummarytable[obssummarytable["filtername"]=="B"]

    fig, (ax1,ax2) = plt.subplots(2,figsize=(8,10),constrained_layout=True)
    fig.suptitle("Zeropoint Calibration Only--No Color Calibration")
    ax1.scatter(obssummarytableB["jd"].value,obssummarytableB["ssr"]/(obssummarytableB["N"]-1),label="B",c="b",s=markersize)
    ax1.scatter(obssummarytableV["jd"].value,obssummarytableV["ssr"]/(obssummarytableV["N"]-1),label="V",c="g",s=markersize)
    ax1.set_xlabel("JD")
    ax1.set_ylabel(r"Photometry Sum Squared Residuals / (N-1) [mag$^2$]")
    ax1.set_yscale("log")
    ax1.grid(True)
    ax1.legend()
    ax2.scatter(obssummarytableB["jd"].value,obssummarytableB["zeropoint"],label="B",c="b",s=markersize)
    ax2.scatter(obssummarytableV["jd"].value,obssummarytableV["zeropoint"],label="V",c="g",s=markersize)
    ax2.set_xlabel("JD")
    ax2.set_ylabel(r"Zeropoint [mag]")
    ax2.grid(True)
    #plt.show()
    fig.savefig(f"CalibNoColorSummary_JD.png")

    fig, (ax1,ax2) = plt.subplots(2,figsize=(8,10),constrained_layout=True)
    fig.suptitle("Zeropoint Calibration Only--No Color Calibration")
    ax1.scatter(obssummarytableB["alt"],obssummarytableB["ssr"]/(obssummarytableB["N"]-1),label="B",c="b",s=markersize)
    ax1.scatter(obssummarytableV["alt"],obssummarytableV["ssr"]/(obssummarytableV["N"]-1),label="V",c="g",s=markersize)
    ax1.set_xlabel("Image Center Altitude [deg]")
    ax1.set_ylabel(r"Photometry Sum Squared Residuals / (N-1) [mag$^2$]")
    ax1.set_yscale("log")
    ax1.grid(True)
    ax1.legend()
    ax2.scatter(obssummarytableB["alt"],obssummarytableB["zeropoint"],label="B",c="b",s=markersize)
    ax2.scatter(obssummarytableV["alt"],obssummarytableV["zeropoint"],label="V",c="g",s=markersize)
    ax2.set_xlabel("Image Center Altitude [deg]")
    ax2.set_ylabel(r"Zeropoint [mag]")
    ax2.grid(True)
    fig.savefig(f"CalibNoColorSummary_Alt.png")

    fig, (ax1,ax2) = plt.subplots(2,figsize=(8,10),constrained_layout=True)
    fig.suptitle("Zeropoint Calibration Only--No Color Calibration")
    ax1.scatter(obssummarytableB["sidereal"],obssummarytableB["ssr"]/(obssummarytableB["N"]-1),label="B",c="b",s=markersize)
    ax1.scatter(obssummarytableV["sidereal"],obssummarytableV["ssr"]/(obssummarytableV["N"]-1),label="V",c="g",s=markersize)
    ax1.set_xlabel("Local Sidereal Time [hourangle]")
    ax1.set_ylabel(r"Photometry Sum Squared Residuals / (N-1) [mag$^2$]")
    ax1.set_yscale("log")
    ax1.grid(True)
    ax1.legend()
    ax2.scatter(obssummarytableB["sidereal"].value,obssummarytableB["zeropoint"],label="B",c="b",s=markersize)
    ax2.scatter(obssummarytableV["sidereal"].value,obssummarytableV["zeropoint"],label="V",c="g",s=markersize)
    ax2.set_xlabel("Local Sidereal Time [hourangle]")
    ax2.set_ylabel(r"Zeropoint [mag]")
    ax2.grid(True)
    fig.savefig(f"CalibNoColorSummary_Sidereal.png")

    fig, (ax1,ax2) = plt.subplots(2,figsize=(8,10),constrained_layout=True)
    fig.suptitle("Zeropoint Calibration Only--No Color Calibration")
    ax1.scatter(obssummarytableB["fwhmpx"],obssummarytableB["ssr"]/(obssummarytableB["N"]-1),label="B",c="b",s=markersize)
    ax1.scatter(obssummarytableV["fwhmpx"],obssummarytableV["ssr"]/(obssummarytableV["N"]-1),label="V",c="g",s=markersize)
    ax1.set_xlabel("FWHM [pixel]")
    ax1.set_ylabel(r"Photometry Sum Squared Residuals / (N-1) [mag$^2$]")
    ax1.set_yscale("log")
    ax1.grid(True)
    ax1.legend()
    ax2.scatter(obssummarytableB["fwhmpx"].value,obssummarytableB["zeropoint"],label="B",c="b",s=markersize)
    ax2.scatter(obssummarytableV["fwhmpx"].value,obssummarytableV["zeropoint"],label="V",c="g",s=markersize)
    ax2.set_xlabel("FWHM [pixel]")
    ax2.set_ylabel(r"Zeropoint [mag]")
    ax2.grid(True)
    fig.savefig(f"CalibNoColorSummary_FWHM.png")

    fig, ax1 = plt.subplots(figsize=(8,10),constrained_layout=True)
    fig.suptitle("Zeropoint Calibration Only--No Color Calibration")
    ax1.scatter(obssummarytableB["zeropoint"],obssummarytableB["ssr"]/(obssummarytableB["N"]-1),label="B",c="b",s=markersize)
    ax1.scatter(obssummarytableV["zeropoint"],obssummarytableV["ssr"]/(obssummarytableV["N"]-1),label="V",c="g",s=markersize)
    ax1.set_xlabel(r"Zeropoint [mag]")
    ax1.set_ylabel(r"Photometry Sum Squared Residuals / (N-1) [mag$^2$]")
    ax1.set_yscale("log")
    ax1.grid(True)
    ax1.legend()
    fig.savefig(f"CalibNoColorSummary_ssrVZeropoint.png")

    ## Add vs FWHM, ellipse measure, and sky background

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
