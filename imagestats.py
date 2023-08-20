#!/usr/bin/env python3

from pathlib import Path
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import astroalign as aa
from ccdproc import Combiner, CCDData

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

printfitsandexit = False

parentdir = Path("~/iTelescope/T18").expanduser()

print(f"Directory: {parentdir}")
for target in parentdir.iterdir():
    if target.name in ["Flats","Darks","Bias"]:
        continue
    if target.suffix == ".pxiproject":
        continue
    print(f"Target: {target.name}")
    fns = list(target.glob("20*/calibrated*-Red-*.fit.zip"))
    stds = []
    print("Finding the lowest noise image...")
    for fn in fns:
        try:
            hdu = fits.open(fn)[0]
            if printfitsandexit:
                for key in hdu.header:
                    print(key,hdu.header[key])
                import sys; sys.exit(0)
            mean, median, std = sigma_clipped_stats(hdu.data,sigma=3.0)
            print(f"{std:.1f} ADU {fn}")
            stds.append(std)
        except OSError as e:
            print(f"OSError: {e}, skipping file: {fn}")
            stds.append(1e20)
    imin = np.argmin(stds)
    ref_data = fits.open(fns[imin])[0].data
    if not ref_data.dtype.isnative:
        ref_data = ref_data.newbyteorder().byteswap()
    print(f"Reference with {stds[imin]:.1f} ADU noise is {fns[imin]}")
    print("Aligning images...")
    aligned_images = []
    for fn in fns:
        this_data = fits.open(fn)[0].data
        if not this_data.dtype.isnative:
            this_data = this_data.newbyteorder().byteswap()
        aligned_data, alightned_footprint = aa.register(this_data,ref_data)
        ccd_data = CCDData(aligned_data,unit="adu")
        aligned_images.append(ccd_data)
    print("Combining images...")
    combiner = Combiner(aligned_images)
    combiner.sigma_clipping(func="median",dev_func="mad_std")
    avg_img = combiner.average_combine()
    avg_fn = f"avg_{target.name}.fit"
    print(f"Saving as {avg_fn}")
    avg_img.write(avg_fn,overwrite=True)
    break
