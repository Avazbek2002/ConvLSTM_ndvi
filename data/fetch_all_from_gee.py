import ee
import geemap
import csv
from datetime import datetime, timedelta
import os # NEW: Import the 'os' module to handle file paths and directories

# --- Initialize Earth Engine and Geemap ---
try:
    # IMPORTANT: Add your Google Cloud Project ID here
    ee.Initialize(project='ndvi-rasters')
    print("Google Earth Engine has been successfully initialized.")
except ee.EEException as e:
    print(f"Authentication or Project setup failed. See error: {e}")
    exit()

# --- 1. Define Area of Interest (AOI) and Parameters ---

geojson_data = {
    "type": "Polygon",
    "coordinates": [[[64.215775, 38.609896],
                     [64.215775, 38.747657],
                     [64.540558, 38.747657],
                     [64.540558, 38.609896],
                     [64.215775, 38.609896]]]}

aoi = ee.Geometry(geojson_data)

# Overall time period
overall_start_date = '2018-01-01'
overall_end_date = '2024-12-31' # Shortened for a quicker test run if needed

# Cloud cover threshold
cloud_filter = 20

# --- CHANGED: Local file configuration ---
# The Google Cloud Storage variables are no longer needed.
# bucket_name = 'ndvi_ndwi_dataset'
# output_folder_gcs = 'raw_images'

# NEW: Define a local folder to save the images
local_output_folder = 'raw_images_local' 
csv_log_file = 'raster_export_log_14Day_local.csv'

# NEW: Create the output directory if it doesn't exist
if not os.path.exists(local_output_folder):
    os.makedirs(local_output_folder)
    print(f"Created local directory: '{local_output_folder}'")


# --- 2. Function to Calculate NDVI and NDWI ---

def calculate_indices(image):
    """Calculates NDVI and NDWI for a Sentinel-2 image."""
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndwi = image.normalizedDifference(['B3', 'B11']).rename('NDWI')
    return image.addBands([ndvi, ndwi])


# --- 3. Generate 14-day Date Intervals ---

date_ranges = []
start = datetime.strptime(overall_start_date, '%Y-%m-%d')
end = datetime.strptime(overall_end_date, '%Y-%m-%d')

while start < end:
    interval_end = start + timedelta(days=14)
    date_ranges.append((
        start.strftime('%Y-%m-%d'),
        interval_end.strftime('%Y-%m-%d')
    ))
    start = interval_end

print(f"Generated {len(date_ranges)} 14-day intervals between {overall_start_date} and {overall_end_date}.")


# --- 4. Load Base Collection and Process Each Interval ---

# Load the Sentinel-2 Harmonized Surface Reflectance collection
s2_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(aoi)

# Open the CSV file to create it and write the header
# CHANGED: Updated the header for local paths
with open(csv_log_file, 'w', newline='') as csvfile:
    log_writer = csv.writer(csvfile)
    log_writer.writerow(['filename', 'interval_start_date', 'local_path'])

    # Loop through each 14-day interval
    for i, (start_date, end_date) in enumerate(date_ranges):
        print(f"\nProcessing interval {i+1}/{len(date_ranges)}: {start_date} to {end_date}")

        # Filter the collection for the current interval
        interval_collection = s2_collection.filterDate(start_date, end_date).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_filter))

        # Check if any images are found in the interval
        if interval_collection.size().getInfo() == 0:
            print("--> No cloud-free images found in this interval. Skipping.")
            continue

        # Create a single seamless mosaic for the interval
        mosaic = interval_collection.mosaic()

        # Calculate indices and select only the final bands
        final_image = calculate_indices(mosaic).select(['NDVI', 'NDWI'])
        
        # Define a descriptive filename
        filename = f"S2_Mosaic_{start_date}.tif"
        
        # NEW: Create the full local file path
        local_filepath = os.path.join(local_output_folder, filename)

        # Log the details to the CSV file
        # CHANGED: Write the local file path to the log
        log_writer.writerow([filename, start_date, local_filepath])

        # --- MAJOR CHANGE: Export the image directly to a local file ---
        print(f"--> Starting download for: {filename} to {local_filepath}")
        task = geemap.ee_export_image_to_drive(
            final_image,
            description=filename,  # This will be the task name in GEE
            folder='GEE_Exports',   # A folder will be created in your Google Drive
            region=aoi,
            scale=10,              # You can keep the high resolution with this method
            fileFormat='GeoTIFF'
        )
        print(f"--> Task submitted for {filename}. Check the 'Tasks' tab in the GEE Code Editor.")

# CHANGED: Updated final confirmation message
print(f"\n✅ All downloads are complete.")
print(f"A log file has been created at '{csv_log_file}'")