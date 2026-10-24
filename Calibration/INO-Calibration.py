# ---------------------------------------------------------------#
# Developed by Leila Sadeghi Ardestani. Last updated 10/24/2026  #
# In case of Inquiries please contact the following email        #
# email address :lsadeghi@ipm.ir                                 #
# github = https://github.com/exoplanetor                        #
# -------------------------------------------------------------- #

from calibration_steps import (
    fix_headers,
    build_hot_pixel_mask,
    apply_hot_pixel_mask,
    run_calibration_pipeline,
    preview_frames,
    inv_median,
    make_master_bias,
    make_master_dark,
    make_dark_corrected_flats,
    make_master_flat_dark_corrected,
    make_master_flat_raw,
    flat_qc,
    build_background_library,
    plot_pipeline_qc
)
from pathlib import Path
import warnings
import logging
from astropy.utils.exceptions import AstropyDeprecationWarning, AstropyUserWarning

# Suppress common Astropy and runtime warnings to keep terminal output clean
warnings.simplefilter('ignore', category=AstropyDeprecationWarning)
warnings.simplefilter('ignore', category=AstropyUserWarning)
warnings.simplefilter('ignore', category=RuntimeWarning)
logging.getLogger('astropy').setLevel(logging.ERROR)
logging.getLogger('ccdproc').setLevel(logging.ERROR)

def ask_yes_no(question):
    while True:
        answer = input(question + " (y/n): ").strip().lower()
        if answer in ["y", "yes"]:
            return True
        if answer in ["n", "no"]:
            return False
        print("Please answer y or n.")


def main():
    folder = input("Enter the folder path with your data: ").strip()
    
    folder_path = Path(folder)    
    if not folder_path.exists():
        print(f"ERROR: Folder does not exist:\n{folder}")
        return    
    if not folder_path.is_dir():
        print(f"ERROR: The path is not a folder:\n{folder}")
        return
    print("You entered:", folder)

    # Step 1: Fix headers
    if ask_yes_no("Do you want to fix headers?"):
        print("Running fix_headers...")
        fix_headers(folder)
        print("Done fixing headers.")

    # Step 2: Hot pixel mask
    if ask_yes_no("Do you want to inspect raw frames?"):
        preview_frames(folder, ask_yes_no)
    
    # Step 3: Hot pixel mask
    if ask_yes_no("Do you want to build hot pixel mask?"):
    
        print("\n-------------------------------------------")
        print("HOT PIXEL MASK REQUIREMENTS")
        print("-------------------------------------------")
        print("Before continuing, make sure that:")
        print("1. You have only TWO Master Dark frames in the mask directory each made up of about 50 dark frames.")
        print("2. Both dark frames have OBSTYPE = DARK in their FITS headers.")
        print("3. The dark frames are located inside the 'mask' folder:")
        print(f"   {folder}\\mask")
        print("4. The two dark frames should have different exposure times")
        print("   (one short exposure and one almost 10 times longer exposure).")
        print("-------------------------------------------\n")
    
        if ask_yes_no("Are these requirements satisfied?"):
            print("Running build_hot_pixel_mask...")
            build_hot_pixel_mask(folder)
            print("Done building hot pixel mask.")
        else:
            print("Skipping hot pixel mask creation.")
    # Step 4: Apply hot pixel mask
    if ask_yes_no("Do you want to apply hot pixel mask?"):
        print("Running apply_hot_pixel_mask...")
        apply_hot_pixel_mask(folder)
        print("Done applying hot pixel mask.")

    # Step 5: Always run full calibration pipeline
    print("Running calibration pipeline...")
    run_calibration_pipeline(folder)
    print("Done calibration pipeline.")


if __name__ == "__main__":
    main()
