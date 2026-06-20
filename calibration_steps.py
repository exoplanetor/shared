# ---------------------------------------------------------------#
# Developed by Leila Sadeghi Ardestani. Last updated 6/6/2026    #
# In case of Inquiries please contact the following email        #
# email address :lsadeghi@ipm.ir                                 #
# github = https://github.com/exoplanetor                        #
# -------------------------------------------------------------- #

import re
from pathlib import Path
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from astropy.nddata import CCDData
from astropy import units as u


# -----------------------------
# Filter detection
# -----------------------------
filter_pattern = re.compile(
    r'(?:^|_|-)(?P<filter>g|i|r|u|clear)(?:_|-|$)',
    re.IGNORECASE
)


# -----------------------------
# Fix headers
# -----------------------------
def fix_headers(main_path):

    path = Path(main_path)

    image_files = [
        f for f in path.rglob("*")
        if f.suffix.lower() in [".fit", ".fits"]
    ]

    for file in image_files:

        name = file.name.lower()

        if "bias" in name:
            obstype = "BIAS"
        elif "dark" in name:
            obstype = "DARK"
        elif "flat" in name:
            obstype = "FLAT"
        else:
            obstype = "OBJECT"

        match = filter_pattern.search(name)
        filter_value = match.group("filter").lower() if match else None

        print(file.name, "->", obstype, filter_value)

        with fits.open(file, mode="update") as hdul:
            hdr = hdul[0].header

            hdr.setdefault("OBSTYPE", obstype)
            hdr.setdefault("BUNIT", "adu")

            if filter_value and "FILTER" not in hdr:
                hdr["FILTER"] = filter_value

            hdul.flush()

# -----------------------------
# Inspect all science images in the directory
# -----------------------------

import os
import glob
import gc
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from astropy.io import fits
from astropy.visualization import simple_norm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def show_scrollable_figure(fig, title="FITS Preview"):

    root = tk.Tk()
    root.title(title)

    canvas = tk.Canvas(root)
    vbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
    hbar = tk.Scrollbar(root, orient="horizontal", command=canvas.xview)

    canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)

    vbar.pack(side="right", fill="y")
    hbar.pack(side="bottom", fill="x")
    canvas.pack(side="left", fill="both", expand=True)

    frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=frame, anchor="nw")

    mpl = FigureCanvasTkAgg(fig, master=frame)
    mpl.draw()
    mpl.get_tk_widget().pack()

    frame.update_idletasks()
    canvas.configure(scrollregion=canvas.bbox("all"))

    def close():
        plt.close(fig)
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close)
    root.mainloop()

    plt.close(fig)


def preview_frames(images_dir, ask_yes_no):

    images = sorted(glob.glob(os.path.join(images_dir, "*.fits")))

    if len(images) == 0:
        print("No FITS files found")
        return False

    groups = {"BIAS": [], "DARK": [], "FLAT": [], "OBJECT": []}

    for f in images:
        try:
            with fits.open(f) as hdul:
                t = hdul[0].header.get("OBSTYPE", "").strip().upper()
                if t in groups:
                    groups[t].append(f)
        except:
            pass

    for obstype, files in groups.items():

        if not files:
            continue

        if not ask_yes_no(f"Inspect {obstype}?"):
            continue

        cols = 3
        rows = (len(files) + cols - 1) // cols

        fig = plt.figure(figsize=(18, 6 * rows))
        fig.suptitle(obstype)

        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.92,
                            wspace=0.4, hspace=0.5)

        for i, f in enumerate(files):

            with fits.open(f) as hdul:
                d = hdul[0].data
                h = hdul[0].header
                d = h.get("BSCALE", 1.0) * d + h.get("BZERO", 0.0)
                d = np.nan_to_num(d)

            ax = plt.subplot(rows, cols, i + 1)
            ax.set_title(os.path.basename(f)[:25], fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

            norm = simple_norm(
                d,
                "sqrt",
                percent=99,
                vmin=np.percentile(d, 5),
                vmax=np.percentile(d, 95),
            )

            ax.imshow(d, norm=norm, cmap="Greys")

            gc.collect()

        show_scrollable_figure(fig, obstype)

    print("Done frame inspection")
    return True
# -----------------------------
# Hot pixel mask builder
# -----------------------------
def build_hot_pixel_mask(main_path):

    main_path = Path(main_path)
    mask_path = main_path / "mask"

    mask_path.mkdir(exist_ok=True)

    dark_files = list(mask_path.glob("*.fit")) + list(mask_path.glob("*.fits"))

    if len(dark_files) < 2:
        print(f"\nFound {len(dark_files)} dark frame(s).")
        print("Please place EXACTLY TWO dark frames in the mask folder.")

        input("Press ENTER when ready...")

        dark_files = list(mask_path.glob("*.fit")) + list(mask_path.glob("*.fits"))

        if len(dark_files) != 2:
            raise RuntimeError(
                f"Expected exactly 2 dark frames, found {len(dark_files)}"
            )

    dark_files = sorted(dark_files)

    dark_short_file, dark_long_file = dark_files

    print("\nUsing:")
    print(f"  Dark_Short: {dark_short_file.name}")
    print(f"  Dark_Long : {dark_long_file.name}")

    dark_short = CCDData.read(dark_short_file)
    dark_long = CCDData.read(dark_long_file)

    # --- inputs ---
    exp_short = float(input("Exposure time Dark_Short (s): "))
    exp_long = float(input("Exposure time Dark_Long (s): "))

    bin_val = input("Binning (1, 2, 3): ").strip()

    bin_map = {
        "1": "1x1",
        "2": "2x2",
        "3": "3x3"
    }

    if bin_val not in bin_map:
        raise ValueError("Binning must be 1, 2, or 3")

    binning_str = bin_map[bin_val]

    # --- convert ---
    dark_short = dark_short.multiply(1.0 * u.electron / u.adu).divide(exp_short * u.second)
    dark_long = dark_long.multiply(1.0 * u.electron / u.adu).divide(exp_long * u.second)

    # People use >1 as a threshold in many detector characterizations. coz Typical CCDs may have dark currents of ~0.001–0.1 e⁻/s at low temperatures, and pixels above 1 e⁻/s are often anomalously high.
    hot_pixels = (dark_long.data > 1)

    # =========================================================
    # 1. DIAGNOSTIC PLOT WITH ITERATIVE AXIS ADJUSTMENT
    # =========================================================

    xlim = (0, 50)
    ylim = (0, 7.5)

    while True:

        plt.figure(figsize=(10, 10))
        plt.plot(
            dark_short.data[hot_pixels].flatten(),
            dark_long.data[hot_pixels].flatten(),
            ".",
            alpha=0.2,
            label="Data"
        )

        plt.xlabel("dark current ($e^-$/sec), short exposure time")
        plt.ylabel("dark current ($e^-$/sec), long exposure time")
        plt.plot([0, 50], [0, 50], label="Ideal relationship x=y")

        plt.xlim(*xlim)
        plt.ylim(*ylim)

        plt.grid()
        plt.legend()
        plt.title("Hot Pixel Diagnostic Plot")

        plt.show()

        adjust = input("\nDo you want to adjust axes? (y/n): ").strip().lower()

        if adjust != "y":
            break

        x_min = float(input("x min: "))
        x_max = float(input("x max: "))
        y_min = float(input("y min: "))
        y_max = float(input("y max: "))

        xlim = (x_min, x_max)
        ylim = (y_min, y_max)

    # =========================================================
    # 2. THRESHOLD SELECTION LOOP
    # =========================================================
# This threshold is found from the plot above and it is where the dark current won't go higher than that value, even if we increaqse the exposure time. it is where they are saturated.
# It is very close to another estimation for hot pixel threshold used by people which is: hot_thresh = median + 1 * std
    
    while True:

        threshold = float(input("\nEnter hot pixel threshold (e-/s): "))

        bad_hot_pixels = (dark_short.data > threshold)

        print("Number of hot pixels detected:", bad_hot_pixels.sum())

        plt.figure(figsize=(5, 5))
        plt.imshow(
            bad_hot_pixels.astype(int),
            cmap="gray",
            origin="lower",
            vmin=0,
            vmax=1,
            interpolation="nearest"
        )
        plt.title("Hot Pixels Mask")
        plt.show()

        ok = input("Are you satisfied with this threshold? (y/n): ").strip().lower()

        if ok == "y":
            break

    # =========================================================
    # 3. SAVE MASK
    # =========================================================

    output_name = (
        f"Mask_hotpixel_thr{threshold:.2f}"
        f"_exp{int(exp_short)}s-{int(exp_long)}s"
        f"_{binning_str}.fits"
    )

    output_file = mask_path / output_name

    CCDData(
        data=bad_hot_pixels.astype("uint8"),
        unit=u.dimensionless_unscaled
    ).write(output_file, overwrite=True)

    print("\nMask saved to:", output_file)


from pathlib import Path
import shutil
import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData
import astropy.units as u

# Apply hot pixel mask to all files in main_path
# -----------------------------
def apply_hot_pixel_mask(main_path):

    main_path = Path(main_path)

    # =========================================================
    # 0. CREATE RAW FOLDER
    # =========================================================
    raw_folder = main_path / "raw"
    raw_folder.mkdir(exist_ok=True)

    # =========================================================
    # 1. AUTO-FIND MASK
    # =========================================================
    mask_path = main_path / "mask"
    
    mask_files = [
        f for f in mask_path.glob("*")
        if f.suffix.lower() in [".fit", ".fits"]
        and "mask" in f.name.lower()
    ]

    if len(mask_files) == 0:
        print("\n⚠ No hot pixel mask found in mask folder.")
        print("\nTo create one, run:")
        print("build_hot_pixel_mask(main_path)")

        create = input("\nDo you want to create a mask now? (y/n): ").strip().lower()

        if create != "y":
            print("Skipping masking step.")
            return

        print("\nPlease create the mask first, then re-run this function.")
        return

    mask_file = sorted(mask_files)[-1]
    print("\nUsing mask:", mask_file.name)

    # =========================================================
    # 2. LOAD MASK
    # =========================================================
    mask_data = fits.getdata(mask_file)
    bad_hot_pixels = mask_data.astype(bool)

    print(f"Mask loaded with {bad_hot_pixels.sum()} hot pixels")

    # =========================================================
    # 3. FIND SCIENCE FILES
    # =========================================================
    all_files = [
        f for f in main_path.glob("*")
        if f.suffix.lower() in [".fit", ".fits"]
        and "mask" not in f.name.lower()
        and "_masked" not in f.name.lower()
    ]

    print(f"\nApplying mask to {len(all_files)} files...")

    # =========================================================
    # 4. APPLY MASK (FIXED)
    # =========================================================
    for file in all_files:

        print("Processing:", file.name)

        data = fits.getdata(file).astype(float)
        header = fits.getheader(file)

        ccdf = CCDData(data, unit=u.adu, meta=header)

        # IMPORTANT: do NOT modify data
        ccdf.mask = bad_hot_pixels

        # Save image + mask as separate HDUs
        primary_hdu = fits.PrimaryHDU(data=ccdf.data, header=ccdf.meta)

        mask_hdu = fits.ImageHDU(
            data=bad_hot_pixels.astype(int),
            name="MASK"
        )

        hdulist = fits.HDUList([primary_hdu, mask_hdu])

        out_file = file.with_name(file.stem + "_masked.fits")

        hdulist.writeto(out_file, overwrite=True)

        print(f"Saved → {out_file.name}")

        # =====================================================
        # MOVE ORIGINAL TO RAW FOLDER
        # =====================================================
        shutil.move(str(file), str(raw_folder / file.name))

        print(f"Moved original → raw/{file.name}")



# import numpy as np
from astropy.io import fits
from pathlib import Path
from astropy.nddata import CCDData
from ccdproc import ImageFileCollection
import ccdproc as ccdp
import matplotlib.pyplot as plt
import astropy.units as u
import glob
from astropy.stats import mad_std
import pandas as pd

def inv_median(ccd):
    med = np.nanmedian(ccd.data)
    return 1.0 / med if med != 0 else 1.0
    
# -----------------------------
# Create Master Bias
# -----------------------------
def make_master_bias(path, combined_path, combined_bias_name):

    combined_bias_file = combined_path / combined_bias_name

    if combined_bias_file.exists():
        print(f"Using existing bias: {combined_bias_file}")
        return CCDData.read(combined_bias_file)

    ifc = ccdp.ImageFileCollection(path)
    all_bias = ifc.files_filtered(OBSTYPE='BIAS', include_path=True)

    if not all_bias:
        raise RuntimeError("No bias frames found.")

    combined_bias = ccdp.combine(
        all_bias,
        method='median',
        sigma_clip=True,
        sigma_clip_low_thresh=5,
        sigma_clip_high_thresh=5,
        sigma_clip_func=np.ma.median,
        sigma_clip_dev_func=mad_std,
        mem_limit=350e6
    )

    combined_bias.meta['combined'] = True
    combined_bias.write(combined_bias_file, overwrite=True)

    print(f"Master bias saved: {combined_bias_file}")

    return combined_bias


# -----------------------------
# Create Master Dark choosing whether or not to include Bias subtraction
# -----------------------------
def make_master_dark(path, combined_path, combined_dark_name, master_bias=None, use_bias=False):

    combined_dark_file = combined_path / combined_dark_name

    if combined_dark_file.exists():
        print(f"Using existing master dark: {combined_dark_file}")
        return CCDData.read(combined_dark_file)

    ifc = ccdp.ImageFileCollection(path)
    all_darks = ifc.files_filtered(OBSTYPE='DARK', include_path=True)

    if not all_darks:
        raise RuntimeError("No dark frames found.")

    dark_list = []

    for f in all_darks:
        ccd = CCDData.read(f, unit='adu')

        # -----------------------------
        # OPTIONAL BIAS SUBTRACTION
        # -----------------------------
        if use_bias:
            if master_bias is None:
                raise ValueError("use_bias=True but no master_bias provided")

            ccd = ccdp.subtract_bias(ccd, master_bias)

        dark_list.append(ccd)

    # -----------------------------
    # COMBINE DARKS
    # -----------------------------
    combined_dark = ccdp.combine(
        dark_list,
        method='average',
        sigma_clip=True,
        sigma_clip_low_thresh=3,
        sigma_clip_high_thresh=3,
        sigma_clip_func=np.ma.median,
        sigma_clip_dev_func=mad_std,
        mem_limit=350e6
    )

    # -----------------------------
    # METADATA
    # -----------------------------
    combined_dark.meta['combined'] = True
    combined_dark.meta['bias_subtracted'] = use_bias

    combined_dark.write(combined_dark_file, overwrite=True)

    print(f"Master dark saved: {combined_dark_file}")

    return combined_dark

# -----------------------------
# Apply darks on Flats
# -----------------------------
def make_dark_corrected_flats(path, combined_path, combined_dark):

    ifc = ccdp.ImageFileCollection(path)

    for ccd, file_name in ifc.ccds(OBSTYPE='FLAT', ccd_kwargs={'unit': 'adu'}, return_fname=True):

        filt = ccd.header.get('FILTER', 'UNKNOWN')

        output_file = combined_path / (
            f'dark_removed_{filt}_{file_name}'
        )

        if output_file.exists():
            continue

        ccd = ccdp.subtract_dark(
            ccd,
            combined_dark,
            exposure_time='exptime',
            exposure_unit=10 * u.microsecond
        )

        ccd.write(output_file, overwrite=True)


def make_master_flat_dark_corrected(combined_path):

    files = glob.glob(str(combined_path / 'dark_removed_*'))

    if not files:
        raise RuntimeError("No dark-corrected flats found")

    flats_by_filter = {}

    for f in files:

        ccd = CCDData.read(f, unit='adu')

        filt = ccd.header.get('FILTER', 'UNKNOWN')

        med = np.nanmedian(ccd.data)

        if med == 0 or np.isnan(med):
            continue

        ccd.data = ccd.data / med

        flats_by_filter.setdefault(filt, []).append(ccd)

    combined_flats = {}

    for filt, flats in flats_by_filter.items():

        output_file = combined_path / f'master_flat_{filt}.fits'

        if output_file.exists():
            combined_flats[filt] = CCDData.read(output_file)
            continue

        combined_flat = ccdp.combine(
            flats,
            method='median',
            sigma_clip=True,
            sigma_clip_low_thresh=5,
            sigma_clip_high_thresh=5,
            sigma_clip_func=np.ma.median,
            sigma_clip_dev_func=mad_std,
            mem_limit=350e6
        )

        combined_flat.meta['combined'] = True

        combined_flat.write(output_file, overwrite=True)

        combined_flats[filt] = combined_flat

    return combined_flats

    
def make_master_flat_raw(path, combined_path):

    files = glob.glob(str(path / 'flat*'))

    if not files:
        raise RuntimeError("No raw flat frames found")

    flats_by_filter = {}

    for f in files:

        ccd = CCDData.read(f, unit='adu')

        filt = ccd.header.get('FILTER', 'UNKNOWN')

        med = np.nanmedian(ccd.data)

        if med == 0 or np.isnan(med):
            continue

        ccd.data = ccd.data / med

        flats_by_filter.setdefault(filt, []).append(ccd)

    combined_flats = {}

    for filt, flats in flats_by_filter.items():

        output_file = combined_path / f'master_flat_raw_{filt}.fits'

        if output_file.exists():
            combined_flats[filt] = CCDData.read(output_file)
            continue

        combined_flat = ccdp.combine(
            flats,
            method='median',
            sigma_clip=True,
            sigma_clip_low_thresh=5,
            sigma_clip_high_thresh=5,
            sigma_clip_func=np.ma.median,
            sigma_clip_dev_func=mad_std,
            mem_limit=350e6
        )

        combined_flat.meta['combined'] = True

        combined_flat.write(output_file, overwrite=True)

        combined_flats[filt] = combined_flat

    return combined_flats


    
# -----------------------------
# Minimal clean QC function for flats
# -----------------------------
def flat_qc(combined_flat, label="Master Flat"):

    data = combined_flat.data.flatten()

    std_value = np.std(data)
    snr = 1 / std_value if std_value != 0 else np.inf

    print(f"\n--- QC: {label} ---")
    print("Std dev:", std_value)
    print("Flat SNR:", snr)

    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=200, histtype='step')
    plt.title(f"Histogram of {label}")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()


# -----------------------------
# BUILD 2D BACKGROUND LIBRARY FUNCTION
# -----------------------------
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_threshold, detect_sources
from photutils.utils import circular_footprint
from astropy.stats import SigmaClip
import numpy as np

def build_background_library(outputs):

    sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
    footprint_radius = 15
    min_pixels = 10

    ny, nx = data.shape
    box_size = (ny // 20, nx // 20)
    filter_size = (3, 3)
    bkg_estimator = MedianBackground()

    plot_order = [
        'raw',
        'minus_dark',
        'minus_dark_div_flat',
        'divide_flat_only',
        'minus_dark_div_non-dark-reduced_flat'
    ]

    masks_dict = {}
    backgrounds_dict = {}
    bgks_dict = {}
    bg_subtracted_dict = {}

    bgk_table = {
        'Data Set': [],
        'Background Median': [],
        'Background RMS Median': []
    }

    for name in plot_order:

        if name not in outputs:
            continue

        data = outputs[name].data

        threshold = detect_threshold(data, nsigma=3, sigma_clip=sigma_clip)
        segment_img = detect_sources(data, threshold, npixels=min_pixels)

        if segment_img is not None:
            footprint = circular_footprint(radius=footprint_radius)
            mask = segment_img.make_source_mask(footprint=footprint)
        else:
            mask = np.zeros_like(data, dtype=bool)

        masks_dict[name] = mask

        bkg = Background2D(
            data,
            box_size=box_size,
            filter_size=filter_size,
            mask=mask,
            sigma_clip=sigma_clip,
            bkg_estimator=bkg_estimator
        )

        bgks_dict[name] = bkg
        backgrounds_dict[name] = bkg.background
        bg_subtracted_dict[name] = data - bkg.background

        bgk_table['Data Set'].append(name)
        bgk_table['Background Median'].append(bkg.background_median)
        bgk_table['Background RMS Median'].append(bkg.background_rms_median)

    df = pd.DataFrame(bgk_table)

    return masks_dict, backgrounds_dict, bgks_dict, bg_subtracted_dict, df, plot_order



def plot_pipeline_qc(outputs, backgrounds_dict, bgks_dict, bg_subtracted_dict, plot_order, title_prefix="QC Check"):

    view_order = [
        'raw',
        'minus_dark_div_flat',
        'minus_dark_div_non-dark-reduced_flat',
        'divide_flat_only'
    ]

    def percent_limits(img, low=1, high=99):
        img = np.asarray(img)
        return np.percentile(img, (low, high))

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    axes = axes.flatten()

    # -----------------------------
    # TOP ROW: calibrated
    # -----------------------------
    for i, name in enumerate(view_order):
        img = outputs[name].data
        vmin, vmax = percent_limits(img)

        axes[i].imshow(img, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        axes[i].set_title(name, fontsize=11)

    # row label (left side)
    axes[0].text(
        -0.6, 0.5,
        "Just Calibrated",
        rotation=90,
        va='center',
        ha='center',
        transform=axes[0].transAxes,
        fontsize=12,
        fontweight='bold'
    )

    # -----------------------------
    # BOTTOM ROW: background-subtracted
    # -----------------------------
    for i, name in enumerate(view_order):
        img = bg_subtracted_dict[name]
        vmin, vmax = percent_limits(img)

        axes[i + 4].imshow(img, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        axes[i + 4].set_title("", fontsize=9)

    # row label (left side)
    axes[4].text(
        -0.6, 0.5,
        "Background subtracted",
        rotation=90,
        va='center',
        ha='center',
        transform=axes[4].transAxes,
        fontsize=12,
        fontweight='bold'
    )

    plt.suptitle(title_prefix, fontsize=14)
    plt.show()

# -----------------------------
# MAIN PIPELINE
# -----------------------------
def run_calibration_pipeline(main_path, selected_mode="1", use_bias=True):

    print("PIPELINE STARTED")
    path = Path(main_path)

    combined_path = path / 'combined'
    calibrated_path = path / 'calibrated'

    combined_path.mkdir(parents=True, exist_ok=True)
    calibrated_path.mkdir(parents=True, exist_ok=True)

    use_bias = input("Include bias subtraction? (y/n): ").strip().lower() == "y"

    combined_bias_name = 'master_bias.fit'
    combined_dark_name = 'master_dark.fits'

    # =========================================================
    # STEP 1 — MASTER CALIBRATION
    # =========================================================
    master_bias = None

    if use_bias:
        master_bias = make_master_bias(
            path,
            combined_path,
            combined_bias_name
        )

    master_dark_path = combined_path / combined_dark_name

    if master_dark_path.exists():
        master_dark = CCDData.read(master_dark_path, unit=u.adu)
    else:
        master_dark = make_master_dark(
            path,
            combined_path,
            combined_dark_name,
            master_bias=master_bias,
            use_bias=use_bias
        )

    # =========================================================
    # BUILD FILTER-SPECIFIC FLATS
    # =========================================================
    make_dark_corrected_flats(
        path,
        combined_path,
        master_dark
    )

    combined_flats = make_master_flat_dark_corrected(
        combined_path
    )

    combined_flats_raw = make_master_flat_raw(
        path,
        combined_path
    )

    # =========================================================
    # OPTIONAL FLAT QC
    # =========================================================
    run_flat_qc = input(
        "\nRun flat QC check for both master flats? (y/n): "
    ).strip().lower()

    if run_flat_qc == "y":

        for filt, flat in combined_flats.items():
            flat_qc(
                flat,
                f"Dark-Corrected Master Flat ({filt})"
            )

        for filt, flat in combined_flats_raw.items():
            flat_qc(
                flat,
                f"Non-Dark-Corrected Master Flat ({filt})"
            )

    # =========================================================
    # SCIENCE INPUTS
    # =========================================================
    print("####################################")
    print(" ")
    print("Cleaning the First Science Frame ...")
    print(" ")
    print("####################################")

    ifc = ccdp.ImageFileCollection(path)

    light_frames = list(
        ifc.files_filtered(
            include_path=True,
            OBSTYPE='object'
        )
    )

    if len(light_frames) == 0:
        raise RuntimeError("No light frames found.")

    # =========================================================
    # STEP 2 — PICK QC FRAME ONLY
    # =========================================================
    qc_frame = light_frames[0]
    qc_raw = CCDData.read(qc_frame, unit=u.adu)
    
    qc_filter = qc_raw.header.get("FILTER")
    
    if qc_filter is None:
        raise ValueError("QC frame has no FILTER keyword")
    
    if qc_filter not in combined_flats:
        raise ValueError(
            f"No dark-corrected master flat found for FILTER={qc_filter}"
        )
    
    if qc_filter not in combined_flats_raw:
        raise ValueError(
            f"No raw master flat found for FILTER={qc_filter}"
        )
    
    exp = qc_raw.header.get('EXPTIME') or qc_raw.header.get('EXPOSURE')
    
    if exp is None:
        raise ValueError("No exposure keyword in QC frame")
    
    
    # =========================================================
    # STEP 3 — CALIBRATE QC FRAME ONLY
    # =========================================================
    qc_minus_dark = ccdp.subtract_dark(
        qc_raw,
        master_dark,
        dark_exposure=float(master_dark.header['EXPTIME']) * u.second,
        data_exposure=float(exp) * u.second,
        exposure_unit=u.second
    )
    
    qc_minus_dark_div_flat = ccdp.flat_correct(
        qc_minus_dark,
        combined_flats[qc_filter]
    )
    
    qc_divide_flat_only = ccdp.flat_correct(
        qc_raw,
        combined_flats_raw[qc_filter]
    )
    
    qc_minus_dark_div_flat2 = ccdp.flat_correct(
        qc_minus_dark,
        combined_flats_raw[qc_filter]
    )
    
    qc_outputs = {
        'raw': qc_raw,
        'minus_dark': qc_minus_dark,
        'minus_dark_div_flat': qc_minus_dark_div_flat,
        'divide_flat_only': qc_divide_flat_only,
        'minus_dark_div_non-dark-reduced_flat': qc_minus_dark_div_flat2
    }

    # =========================================================
    # STEP 4 — BACKGROUND ONLY ON QC FRAME
    # =========================================================
    masks_dict, backgrounds_dict, bgks_dict, bg_subtracted_dict, *_ = (
        build_background_library(
            outputs=qc_outputs
        )
    )

    # =========================================================
    # STEP 4.5 — SAVE QC PRODUCTS
    # =========================================================
    qc_save_map = {
        "minus_dark_div_flat": qc_outputs["minus_dark_div_flat"],
        "minus_dark_div_non-dark-reduced_flat": qc_outputs["minus_dark_div_non-dark-reduced_flat"],
        "divide_flat_only": qc_outputs["divide_flat_only"]
    }

    for name, ccd in qc_save_map.items():

        if ccd.meta is None:
            print(f"WARNING: {name} has no metadata!")

        out_path = calibrated_path / f"QC_{name}.fits"

        ccd.write(out_path, overwrite=True)

        print(f"Saved QC product: {out_path.name}")

    for name, img in bg_subtracted_dict.items():

        source_ccd = qc_outputs.get(name)

        if source_ccd is None:
            raise ValueError(
                f"No source CCD found for {name}"
            )

        ccd = CCDData(
            img,
            unit=source_ccd.unit,
            meta=source_ccd.meta.copy()
        )

        out_path = calibrated_path / f"QC_{name}_bgsub.fits"

        ccd.write(out_path, overwrite=True)

        print(
            f"Saved QC background-subtracted: "
            f"{out_path.name}"
        )

    # =========================================================
    # STEP 5 — VIEW ORDER
    # =========================================================
    print("####################################")
    print(" ")
    print("Preparing the Plots ...")
    print(" ")
    print("####################################")

    view_order = [
        'raw',
        'minus_dark',
        'minus_dark_div_flat',
        'divide_flat_only',
        'minus_dark_div_non-dark-reduced_flat'
    ]

    # =========================================================
    # STEP 6 — SHOW QC PLOTS
    # =========================================================
    plot_pipeline_qc(
        qc_outputs,
        backgrounds_dict,
        bgks_dict,
        bg_subtracted_dict,
        view_order,
        title_prefix="QC Check"
    )

    # =========================================================
    # STEP 7 — USER CHOICE
    # =========================================================
    print("\nChoose reduction:")
    print("1: minus_dark_div_flat")
    print("2: minus_dark_div_non-dark-reduced_flat")
    print("3: divide_flat_only")

    choice = input("Enter choice: ").strip()

    if choice == "1":
        selected_mode = "minus_dark_div_flat"
    elif choice == "2":
        selected_mode = "minus_dark_div_non-dark-reduced_flat"
    elif choice == "3":
        selected_mode = "divide_flat_only"
    else:
        raise ValueError("Invalid choice")

    apply_bg = (
        input(
            "\nApply Background2D globally to all frames? (y/n): "
        ).strip().lower() == "y"
    )

# =========================================================
# STEP 8 — APPLY TO ALL FRAMES
# =========================================================
for frame in light_frames:

    raw = CCDData.read(frame, unit=u.adu)

    mask_hdu = fits.open(frame)['MASK'].data
    raw.mask = mask_hdu.astype(bool)

    filt = raw.header.get("FILTER")

    if filt is None:
        raise ValueError(f"No FILTER keyword in {frame}")

    if filt not in combined_flats:
        raise ValueError(
            f"No dark-corrected flat for FILTER={filt}"
        )

    if filt not in combined_flats_raw:
        raise ValueError(
            f"No raw flat for FILTER={filt}"
        )

    exp = raw.header.get('EXPTIME') or raw.header.get('EXPOSURE')

    if exp is None:
        raise ValueError(f"No exposure keyword in {frame}")

    minus_dark = ccdp.subtract_dark(
        raw,
        master_dark,
        dark_exposure=float(master_dark.header['EXPTIME']) * u.second,
        data_exposure=float(exp) * u.second,
        exposure_unit=u.second
    )

    # -----------------------------
    # FLAT SELECTION (FILTER-SAFE)
    # -----------------------------
    if selected_mode == "minus_dark_div_flat":

        final = ccdp.flat_correct(
            minus_dark,
            combined_flats[filt]
        )

    elif selected_mode == "minus_dark_div_non-dark-reduced_flat":

        final = ccdp.flat_correct(
            minus_dark,
            combined_flats_raw[filt]
        )

    else:

        final = ccdp.flat_correct(
            raw,
            combined_flats_raw[filt]
        )

    data = final.data

    if apply_bg:

        threshold = detect_threshold(data, nsigma=3)

        segment_img = detect_sources(data, threshold, npixels=10)

        mask = segment_img.make_source_mask() if segment_img else None

        bkg = Background2D(
            data,
            (50, 50),
            mask=mask,
            bkg_estimator=MedianBackground()
        )

        data = data - bkg.background

    final_ccd = CCDData(
        data,
        header=final.header,
        unit=final.unit
    )

    suffix = "_bg" if apply_bg else ""

    output_name = (
        f"{Path(frame).stem}_"
        f"{selected_mode}"
        f"{suffix}_final.fits"
    )

    final_ccd.write(
        calibrated_path / output_name,
        overwrite=True
    )

    print(f"Saved: {output_name}")

print("\nPipeline complete.\n")

return combined_flats, combined_flats_raw
