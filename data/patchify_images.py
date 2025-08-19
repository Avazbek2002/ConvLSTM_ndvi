import os
import glob
import re
import rasterio
from rasterio.windows import Window


def create_georeferenced_patches(input_dir, output_dir, patch_size=256):
    """
    Processes GeoTIFF images, creating a dated folder for each, and saves
    georeferenced square patches. This version preserves the geospatial
    metadata (CRS and transform) for each patch.

    Args:
        input_dir (str): The directory containing the time-series GeoTIFF images.
        output_dir (str): The directory where the dated folders with patches will be saved.
        patch_size (int, optional): The size of the square patches. Defaults to 256.
    """
    # Ensure the main output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Find all TIFF files
    tiff_files = glob.glob(os.path.join(input_dir, '*.tif*'))

    if not tiff_files:
        print(f"⚠️ No TIFF files found in the directory: {input_dir}")
        return

    print(f"Found {len(tiff_files)} TIFF files to process.")

    for tiff_file in tiff_files:
        try:
            with rasterio.open(tiff_file) as src:
                # --- 1. Extract date and create folder ---
                image_date = None
                match = re.search(r'\d{4}-\d{2}-\d{2}', os.path.basename(tiff_file))
                if match:
                    image_date = match.group(0)
                else:
                    print(f"❌ Could not extract date for: {os.path.basename(tiff_file)}. Skipping.")
                    continue
                
                date_folder = os.path.join(output_dir, image_date)
                os.makedirs(date_folder, exist_ok=True)
                
                width = src.width
                height = src.height
                
                print(f"Processing {os.path.basename(tiff_file)} (Date: {image_date}, Size: {width}x{height})")

                # --- 2. Manually create and save georeferenced patches ---
                patch_count = 0
                for y in range(0, height, patch_size):
                    for x in range(0, width, patch_size):
                        # Create patches only from full-sized areas
                        if (y + patch_size > height) or (x + patch_size > width):
                            continue

                        # Define the window to read from the source image
                        window = Window(x, y, patch_size, patch_size)

                        # Get the transform for the new patch window
                        # This calculates the new top-left coordinate for the patch
                        new_transform = src.window_transform(window)

                        # Read the data from the window
                        patch_data = src.read(window=window)

                        # Get a copy of the source metadata and update it for the patch
                        profile = src.profile.copy()
                        profile.update({
                            'height': patch_size,
                            'width': patch_size,
                            'transform': new_transform
                        })
                        
                        # Define a unique filename and save the patch
                        patch_row = y // patch_size
                        patch_col = x // patch_size
                        patch_filename = f"patch_{patch_row}_{patch_col}.tif"
                        patch_filepath = os.path.join(date_folder, patch_filename)
                        
                        # Write the new georeferenced patch using rasterio
                        with rasterio.open(patch_filepath, 'w', **profile) as dst:
                            dst.write(patch_data)
                        
                        patch_count += 1
                
                print(f"✅ Saved {patch_count} georeferenced patches to {date_folder}")

        except Exception as e:
            print(f"❌ Could not process file {os.path.basename(tiff_file)}. Error: {e}")


if __name__ == '__main__':
    # --- How to use the function ---

    # 1. Set the directory containing your GeoTIFF images
    input_directory = '/Users/isroilov/Desktop/Pasture/rasters'

    # 2. Set the directory where you want to save the patched images
    output_directory = '/Users/isroilov/Desktop/Pasture/patched'

    # 3. Set the desired size for the square patches (e.g., 256x256 pixels)
    patch_size_px = 64

    # 4. Call the function to process the images
    create_georeferenced_patches(input_directory, output_directory, patch_size_px)

    print("\n🚀 Processing complete.")


