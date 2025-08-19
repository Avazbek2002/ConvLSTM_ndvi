import os
import glob
from datetime import datetime, timedelta
import rasterio
import numpy as np

# --- Configuration ---
# Folder containing your original TIFF images
INPUT_DIR = '/Users/isroilov/Desktop/Pasture/GEE_Exports'
# Folder where the new interpolated images will be saved
OUTPUT_DIR = '/Users/isroilov/Desktop/Pasture/patched'
# --- End of Configuration ---

def parse_date_from_filename(filename):
    """Extracts the date from a filename like 'S2_Mosaic_YYYY-MM-DD.tif'."""
    base_name = os.path.basename(filename)
    date_str = base_name.split('_')[-1].replace('.tif', '')
    return datetime.strptime(date_str, '%Y-%m-%d')

def interpolate_rasters():
    """Finds large time gaps in a raster series and fills them with interpolation."""
    
    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get a list of all TIFF files in the input directory
    tiff_files = glob.glob(os.path.join(INPUT_DIR, '*.tif'))
    
    if len(tiff_files) < 2:
        print("Need at least two images to perform interpolation. Exiting.")
        return

    # Sort the files chronologically based on their date
    sorted_files = sorted(tiff_files, key=parse_date_from_filename)
    print(f"Found and sorted {len(sorted_files)} images.")

    # Loop through consecutive pairs of images
    for i in range(len(sorted_files) - 1):
        file1_path = sorted_files[i]
        file2_path = sorted_files[i+1]
        
        date1 = parse_date_from_filename(file1_path)
        date2 = parse_date_from_filename(file2_path)
        
        # Calculate the time difference in days
        time_gap = date2 - date1
        
        print(f"\nComparing '{os.path.basename(file1_path)}' and '{os.path.basename(file2_path)}'...")
        print(f"Time gap: {time_gap.days} days.")
        
        # Check if the gap is larger than 15 days
        if time_gap.days > 15:
            print("--> Gap is > 15 days. Performing linear interpolation.")
            
            # Open the two images
            with rasterio.open(file1_path) as src1, rasterio.open(file2_path) as src2:
                # Read image data into numpy arrays
                array1 = src1.read()
                array2 = src2.read()
                
                # Get the metadata (profile) from the first image to use for the new one
                profile = src1.profile
                
                # Perform linear interpolation (pixel-wise average)
                # This assumes both images have the same shape and number of bands
                interpolated_array = (array1 + array2) / 2
                
                # Calculate the date for the new interpolated image (midpoint)
                midpoint_date = date1 + time_gap / 2
                
                # Create a filename for the new image
                new_filename = f"S2_Mosaic_{midpoint_date.strftime('%Y-%m-%d')}.tif"
                output_path = os.path.join(OUTPUT_DIR, new_filename)
                
                # Ensure the output data type matches the profile
                interpolated_array = interpolated_array.astype(profile['dtype'])
                
                # Write the new raster file
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(interpolated_array)
                    
                print(f"--> ✅ Created interpolated image: {new_filename}")
        else:
            print("--> Gap is <= 15 days. Skipping.")
            
    print("\nProcessing complete.")

if __name__ == '__main__':
    interpolate_rasters()