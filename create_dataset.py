import os
import shutil
import requests
import math
import shapefile
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from rtree import index
from shapely import Polygon
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

DEBUG = False

if DEBUG:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    debug_data = []  # Store data for debugging


OUTPUT_DIR = "dataset"

# DOP WMS service URL and layer name
WMS_URL = "https://www.wms.nrw.de/geobasis/wms_nw_dop"
LAYER_NAME = "nw_dop_rgb"
WMS_WORKERS = 5
LAYER_GSD = 0.1     # m/pixel

TARGET_GSD = 0.2    # m/pixel

# Directory to store images
DOP_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "dop_images")
os.makedirs(DOP_OUTPUT_DIR, exist_ok=True)

# SAM encoder input size
IMG_WIDTH = 1024
IMG_HEIGHT = 1024

# GRU source URL for Wuppertal
GRU_URL = "https://www.opengeodata.nrw.de/produkte/geobasis/lk/akt/gru_xml/gru_05124000_Wuppertal_EPSG25832_NAS.zip"
GRU_DOWNLOAD_PATH = os.path.join(OUTPUT_DIR, "gru.zip") 
GRU_EXTRACT_PATH = os.path.join(OUTPUT_DIR, "gru") 
OGR_OUTPUT = os.path.join(OUTPUT_DIR, "roof_shape") 
os.makedirs(OGR_OUTPUT, exist_ok=True)
OGR_LAYER = "AX_Gebaeude"
OGR_SRS = "EPSG:25832"
OGR_WHERE = "dachform IS NOT NULL"

def download_gru_zip():
    """Downloads the GRU ZIP file using wget."""
    if not os.path.exists(GRU_DOWNLOAD_PATH):
        os.system(f"wget -O {GRU_DOWNLOAD_PATH} {GRU_URL}")
        print("GRU data downloaded.")
    else:
        print("File already exists, skipping download.")

def extract_relevant_file(search_string):
    """Finds and extracts only the file containing the search string from the ZIP archive."""
    file_list = os.popen(f"unzip -l {GRU_DOWNLOAD_PATH}").read()
    relevant_file = None

    for line in file_list.split("\n"):
        if search_string in line:
            relevant_file = line.split()[-1]
            break

    if relevant_file:
        os.system(f"unzip -o {GRU_DOWNLOAD_PATH} {relevant_file} -d {GRU_EXTRACT_PATH}")
        print(f"Extracted relevant file: {relevant_file}")
        return os.path.join(GRU_EXTRACT_PATH, relevant_file)
    else:
        print("No relevant file found.")
        return None

def convert_gru_to_shapefile(xml_file, ogr_output_shp):
    """Converts the extracted XML file to an ESRI Shapefile using ogr2ogr."""
    if xml_file:
        os.system(f"ogr2ogr -f \"ESRI Shapefile\" {ogr_output_shp} {xml_file} {OGR_LAYER}"
                  " -s_srs {OGR_SRS} -t_srs {OGR_SRS} -where \"{OGR_WHERE}\"")
        if os.path.exists(ogr_output_shp):
            print(f"Converted {xml_file} to {ogr_output_shp}")
        else:
            print("File conversion failed.")

def fetch_and_save_dop(tile, tile_index):
    """Fetches and saves a DOP image for the given tile."""

    file_name = f"tile_{tile_index}.png"
    output_path = os.path.join(DOP_OUTPUT_DIR, file_name)

    # skip if file exists
    if os.path.exists(output_path):
        return None

    (tile_min_x, tile_min_y), (tile_max_x, tile_max_y) = tile
    bbox = f"{tile_min_x},{tile_min_y},{tile_max_x},{tile_max_y}"

    params = {
        "service": "WMS",
        "request": "GetMap",
        "version": "1.3.0",
        "layers": LAYER_NAME,
        "styles": "",
        "crs": "EPSG:25832",
        "bbox": bbox,
        "width": IMG_WIDTH,
        "height": IMG_HEIGHT,
        "format": "image/png"
    }

    response = requests.get(WMS_URL, params=params, timeout=5)

    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        image.save(output_path)
        print(f"Saved: {output_path}")
        return file_name
    else:
        print(f"Error fetching tile {tile_index}: HTTP {response.status_code}")
        return None

def get_min_max_coordinates(shapefile_path):
    """Reads the min and max coordinates from a shapefile."""
    with shapefile.Reader(shapefile_path) as sf:
        min_x, min_y, max_x, max_y = float("inf"), float("inf"), float("-inf"), float("-inf")

        for shape in sf.shapes():
            if shape.shapeTypeName == "POLYGON":
                x_coords, y_coords = zip(*shape.points)
                min_x, max_x = min(min_x, min(x_coords)), max(max_x, max(x_coords))
                min_y, max_y = min(min_y, min(y_coords)), max(max_y, max(y_coords))

        return min_x, min_y, max_x, max_y

def generate_tiles(min_x, min_y, max_x, max_y, tile_width, tile_height):
    """Generates tiles based on bounding coordinates and tile size."""
    tiles = []

    x_range = max_x - min_x
    y_range = max_y - min_y

    num_tiles_x = math.ceil(x_range / tile_width)
    num_tiles_y = math.ceil(y_range / tile_height)

    for i in range(num_tiles_x):
        for j in range(num_tiles_y):
            tile_min_x = min_x + i * tile_width
            tile_max_x = tile_min_x + tile_width
            tile_min_y = min_y + j * tile_height
            tile_max_y = tile_min_y + tile_height

            tiles.append(((tile_min_x, tile_min_y), (tile_max_x, tile_max_y)))

    return tiles

def filter_tiles_with_rtree(tiles, shapefile_path, tile_width, tile_height):
    """Filters tiles that FULLY contain at least one feature using an R-Tree index."""
    spatial_index = index.Index()

    shapes = []

    out_list = []

    match_count = 0

    to_pixel_width = IMG_WIDTH / tile_width
    to_pixel_height = IMG_HEIGHT / tile_height

    with shapefile.Reader(shapefile_path) as sf:
        shapes = sf.shapes()
        records = sf.records()
        for i, shape in enumerate(shapes):
            if shape.shapeTypeName == "POLYGON":
                x_coords, y_coords = zip(*shape.points)
                spatial_index.insert(i, (min(x_coords), min(y_coords),
                                          max(x_coords), max(y_coords)))

    for tile in tiles:
        (tile_min_x, tile_min_y), (tile_max_x, tile_max_y) = tile
        possible_matches = list(
            spatial_index.contains((tile_min_x, tile_min_y, tile_max_x, tile_max_y))
            )

        if possible_matches:
            matches = []
            for match in possible_matches:
                matched_shape = shapes[match]
                matched_record = records[match]
                x_coords, y_coords = zip(*matched_shape.points)

                # transform shape to pixel coordinates
                x_coords_pixel = [(x - tile_min_x) * to_pixel_width for x in x_coords]
                y_coords_pixel = [(y - tile_min_y) * -to_pixel_height + IMG_HEIGHT 
                                  for y in y_coords]

                shape_points = zip(x_coords_pixel, y_coords_pixel)
                shape_centroid = Polygon(shape_points).centroid
                matches.append({"points": [(x_coords_pixel[i], y_coords_pixel[i])
                                           for i in range(len(x_coords_pixel))],
                                "centroid": (shape_centroid.x, shape_centroid.y),
                                "class": matched_record["dachform"]})

                match_count += 1


            out_list.append({"tile": tile,
                             "contained_polygons": matches})

    print(f"Found {match_count} roof polygons that are fully contained in {len(out_list)} tiles.")

    return out_list

def process_tile(index, tile_and_shapes):
    """Function to fetch a DOP image and update the dictionary."""
    tile = tile_and_shapes["tile"]
    dop_name = fetch_and_save_dop(tile, index)
    tile_and_shapes["dop_name"] = dop_name

    if DEBUG:
        debug_data.append((dop_name, tile_and_shapes["contained_polygons"]))

    return dop_name

def plot_tile_and_shapes(file_name, polygons):
    """
    Plots a tile image along with the overlaid polygons representing buildings.
    
    :param file_name: Name of the tile image file.
    :param polygons: List of dictionaries containing polygon points and centroids.
    """
    file_path = os.path.join(DOP_OUTPUT_DIR, file_name)

    if not os.path.exists(file_path):
        print(f"Image file {file_path} not found.")
        return

    # Load the image
    image = Image.open(file_path)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))

    # Inverted y-axis to match image coordinates
    ax.imshow(image, extent=[0, IMG_WIDTH, IMG_HEIGHT, 0])

    # Plot polygons
    for polygon in polygons:
        points = polygon["points"]
        polygon_patch = patches.Polygon(
            points, closed=True, edgecolor='red', facecolor='none', linewidth=2
            )
        ax.add_patch(polygon_patch)

        # Plot centroid
        centroid_x, centroid_y = polygon["centroid"]
        ax.plot(centroid_x, centroid_y, 'bo', markersize=5, label="Centroid")

    # Display the plot
    plt.axis("off")
    plt.show(block=True)

def delete_path(path):
    """Deletes a file or a folder (including subfolders) if the path exists."""
    if not os.path.exists(path):
        print(f"Path does not exist: {path}")
        return
   
    if os.path.isfile(path):
        os.remove(path)
        print(f"Deleted file: {path}")
    elif os.path.isdir(path):
        shutil.rmtree(path)
        print(f"Deleted folder and its contents: {path}")
    else:
        print(f"Unknown type: {path}")

def cleanup():
    delete_path(GRU_DOWNLOAD_PATH)
    delete_path(GRU_EXTRACT_PATH)
    delete_path(OGR_OUTPUT)
    return

if __name__ == "__main__":
    # Download and process GRU data
    download_gru_zip()
    file_found = extract_relevant_file("31001") # key for "AX_Gebaeude"
    ogr_output_shp = os.path.join(OGR_OUTPUT, "output_converted.shp")
    if file_found:
        print(f"File containing 'AX_Gebaeude' extracted: {file_found}")
        print("Extracting roof shapes and converting to ESRI shape file")
        convert_gru_to_shapefile(file_found, ogr_output_shp)
    else:
        print("No file containing 'AX_Gebaeude' was found.")

    min_x, min_y, max_x, max_y = get_min_max_coordinates(ogr_output_shp)
    tiles = generate_tiles(
        min_x, min_y, max_x, max_y, IMG_WIDTH * TARGET_GSD, IMG_HEIGHT * TARGET_GSD
        )

    filtered_shapes = filter_tiles_with_rtree(
        tiles, ogr_output_shp, IMG_WIDTH * TARGET_GSD, IMG_HEIGHT * TARGET_GSD
        )

    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor(max_workers=WMS_WORKERS) as executor:  # Adjust workers as needed
        futures = {executor.submit(process_tile, i, tile): tile 
                   for i, tile in enumerate(filtered_shapes)}

        for future in tqdm(as_completed(futures), total=len(filtered_shapes)):
            dop_name = future.result()

    print("Saving metadata")
    metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(metadata_path, 'w') as fp:
        json.dump(filtered_shapes, fp)

    # Plot debug images sequentially (to avoid threading issues)
    if DEBUG:
        for dop_name, polygons in debug_data:
            plot_tile_and_shapes(dop_name, polygons)

    cleanup()
