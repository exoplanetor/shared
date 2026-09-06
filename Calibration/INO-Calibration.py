# ---------------------------------------------------------------#
# Developed by Leila Sadeghi Ardestani. Last updated 6/6/2026    #
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
        print("Running build_hot_pixel_mask...")
        build_hot_pixel_mask(folder)
        print("Done building hot pixel mask.")

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
