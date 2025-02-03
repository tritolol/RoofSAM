import os
import requests
import math
import shapefile
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from rtree import index

OUTPUT_DIR = "dataset"

# DOP WMS service URL and layer name
WMS_URL = "https://www.wms.nrw.de/geobasis/wms_nw_dop"
LAYER_NAME = "nw_dop_rgb"
LAYER_GSD = 0.1     # m/pixel

# Directory to store images
DOP_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "dop_images")
os.makedirs(DOP_OUTPUT_DIR, exist_ok=True)

# SAM encoder input size
IMG_WIDTH = 1024
IMG_HEIGHT = 1024

# GRU source URL for Wuppertal
GRU_URL = "https://www.opengeodata.nrw.de/produkte/geobasis/lk/akt/gru_xml/gru_05124000_Wuppertal_EPSG25832_NAS.zip"
GRU_DOWNLOAD_PATH = os.path.join(OUTPUT_DIR, "gru_wuppertal.zip") 
GRU_EXTRACT_PATH = os.path.join(OUTPUT_DIR, "gru_wuppertal") 
OGR_OUTPUT_SHP = os.path.join(OUTPUT_DIR, "converted_output.shp") 
OGR_INPUT_XML = "W_fachinformationen_31001_gru.xml"
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

def convert_gru_to_shapefile(xml_file):
    """Converts the extracted XML file to an ESRI Shapefile using ogr2ogr."""
    if xml_file:
        os.system(f"ogr2ogr -f \"ESRI Shapefile\" {OGR_OUTPUT_SHP} {xml_file} {OGR_LAYER} -s_srs {OGR_SRS} -t_srs {OGR_SRS} -where \"{OGR_WHERE}\"")
        if os.path.exists(OGR_OUTPUT_SHP):
            print(f"Converted {xml_file} to {OGR_OUTPUT_SHP}")
        else:
            print("File conversion failed.")

def fetch_and_save_dop(tile, tile_index):
    """Fetches and saves a DOP image for the given tile."""
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
        output_path = os.path.join(DOP_OUTPUT_DIR, f"tile_{tile_index}.png")
        image.save(output_path)
        print(f"Saved: {output_path}")
    else:
        print(f"Error fetching tile {tile_index}: HTTP {response.status_code}")

def get_min_max_coordinates(shapefile_path):
    """Reads the min and max coordinates from a shapefile."""
    with shapefile.Reader(shapefile_path) as sf:
        min_x, min_y, max_x, max_y = float("inf"), float("inf"), float("-inf"), float("-inf")

        for shape in sf.shapes():
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

def filter_tiles_with_rtree(tiles, shapefile_path):
    """Filters tiles that contain at least one feature using an R-Tree index."""
    spatial_index = index.Index()

    with shapefile.Reader(shapefile_path) as sf:
        for i, shape in enumerate(sf.shapes()):
            x_coords, y_coords = zip(*shape.points)
            spatial_index.insert(i, (min(x_coords), min(y_coords), max(x_coords), max(y_coords)))

    valid_tiles = []
    for tile in tiles:
        (tile_min_x, tile_min_y), (tile_max_x, tile_max_y) = tile
        possible_matches = list(spatial_index.intersection((tile_min_x, tile_min_y, tile_max_x, tile_max_y)))

        if possible_matches:
            valid_tiles.append(tile)

    return valid_tiles



if __name__ == "__main__":
    # Download and process GRU data
    download_gru_zip()
    file_found = extract_relevant_file("31001") # key for "AX_Gebaeude"
    if file_found:
        print(f"File containing 'AX_Gebaeude' extracted: {file_found}")
        print("Extracting roof shapes and converting to ESRI shape file")
        convert_gru_to_shapefile(file_found)
    else:
        print("No file containing 'AX_Gebaeude' was found.")

    min_x, min_y, max_x, max_y = get_min_max_coordinates(OGR_OUTPUT_SHP)
    tiles = generate_tiles(min_x, min_y, max_x, max_y, IMG_WIDTH * LAYER_GSD, IMG_HEIGHT * LAYER_GSD)

    filtered_tiles = filter_tiles_with_rtree(tiles, OGR_OUTPUT_SHP)

    for index, tile in enumerate(tqdm(filtered_tiles)):
        fetch_and_save_dop(tile, index)