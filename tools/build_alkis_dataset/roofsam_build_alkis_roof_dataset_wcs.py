#!/usr/bin/env python3
import os
import shutil
import math
import json
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import requests
import shapefile
from tqdm import tqdm
from PIL import Image
from rtree import index
from shapely import Polygon

# Optional debug plotting
DEBUG = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and process DOP images and roof polygon data using a WCS service."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset",
        help="Directory to store outputs. Default is 'dataset'.",
    )
    parser.add_argument(
        "--wcs-url",
        type=str,
        default="https://www.wcs.nrw.de/geobasis/wcs_nw_dop",
        help="WCS service URL for DOP data. Default is https://www.wcs.nrw.de/geobasis/wcs_nw_dop",
    )
    parser.add_argument(
        "--layer-name",
        type=str,
        default="nw_dop",
        help="WCS layer name (coverageId) to request. Default is 'nw_dop'.",
    )
    parser.add_argument(
        "--wcs-workers",
        type=int,
        default=5,
        help="Number of worker threads for fetching WCS tiles.",
    )
    # Note: SCALEFACTOR is computed as layer_gsd/target_gsd.
    parser.add_argument(
        "--layer-gsd",
        type=float,
        default=0.1,
        help="Native ground sampling distance (m/pixel) of the DOP layer. For example, 0.1.",
    )
    parser.add_argument(
        "--target-gsd",
        type=float,
        default=0.1,
        help="Desired ground sampling distance (m/pixel) for processing tiles. For example, 1.0.",
    )
    parser.add_argument(
        "--img-width",
        type=int,
        default=1024,
        help="Image width in pixels. Default is 1024.",
    )
    parser.add_argument(
        "--img-height",
        type=int,
        default=1024,
        help="Image height in pixels. Default is 1024.",
    )
    parser.add_argument(
        "--gru-url",
        type=str,
        default="https://www.opengeodata.nrw.de/produkte/geobasis/lk/akt/gru_xml/gru_05124000_Wuppertal_EPSG25832_NAS.zip",
        help="URL to download GRU data.",
    )
    parser.add_argument(
        "--ogr-layer",
        type=str,
        default="AX_Gebaeude",
        help="OGR layer name to extract from the GRU data.",
    )
    parser.add_argument(
        "--ogr-srs",
        type=str,
        default="EPSG:25832",
        help="Spatial reference system to use with ogr2ogr.",
    )
    parser.add_argument(
        "--ogr-where",
        type=str,
        default="dachform IS NOT NULL",
        help="SQL WHERE clause to filter features during ogr2ogr conversion.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="If set, enables debug mode with plotting.",
    )
    return parser.parse_args()


def download_gru_zip(gru_url, gru_download_path):
    """Downloads the GRU ZIP file using wget."""
    if not os.path.exists(gru_download_path):
        os.system(f"wget -O {gru_download_path} {gru_url}")
        print("GRU data downloaded.")
    else:
        print("GRU ZIP file already exists, skipping download.")


def extract_relevant_file(gru_download_path, gru_extract_path, search_string):
    """Finds and extracts only the file containing the search string from the ZIP archive."""
    file_list = os.popen(f"unzip -l {gru_download_path}").read()
    relevant_file = None

    for line in file_list.split("\n"):
        if search_string in line:
            relevant_file = line.split()[-1]
            break

    if relevant_file:
        os.makedirs(gru_extract_path, exist_ok=True)
        os.system(f"unzip -o {gru_download_path} {relevant_file} -d {gru_extract_path}")
        print(f"Extracted relevant file: {relevant_file}")
        return os.path.join(gru_extract_path, relevant_file)
    else:
        print("No relevant file found in the GRU ZIP archive.")
        return None


def convert_gru_to_shapefile(xml_file, ogr_output_shp, ogr_layer, ogr_srs, ogr_where):
    """Converts the extracted XML file to an ESRI Shapefile using ogr2ogr."""
    if xml_file:
        os.system(
            f'ogr2ogr -f "ESRI Shapefile" {ogr_output_shp} {xml_file} {ogr_layer} '
            f'-s_srs {ogr_srs} -t_srs {ogr_srs} -where "{ogr_where}"'
        )
        if os.path.exists(ogr_output_shp):
            print(f"Converted {xml_file} to {ogr_output_shp}")
        else:
            print("File conversion failed.")


def fetch_and_save_dop(
    tile,
    tile_index,
    dop_output_dir,
    wcs_url,
    layer_name,
    img_width,
    img_height,
    target_gsd,
    layer_gsd,
):
    """
    Fetches and saves a DOP image for the given tile using a WCS GetCoverage request.

    The request URL is built with parameters similar to:
      ?VERSION=2.0.1&SERVICE=wcs&REQUEST=GetCoverage&COVERAGEID=nw_dop&FORMAT=image/tiff
      &SUBSET=x(min,max)&SUBSET=y(min,max)&SCALEFACTOR=...

    For example:
      https://www.wcs.nrw.de/geobasis/wcs_nw_dop?VERSION=2.0.1&SERVICE=wcs&REQUEST=GetCoverage&COVERAGEID=nw_dop&FORMAT=image/tiff
      &SUBSET=x(304000,304100)&SUBSET=y(5632000,5632100)&SCALEFACTOR=0.1
    """
    file_name = f"tile_{tile_index}.png"
    output_path = os.path.join(dop_output_dir, file_name)

    # Skip if file exists
    if os.path.exists(output_path):
        return file_name

    (tile_min_x, tile_min_y), (tile_max_x, tile_max_y) = tile

    # Compute the scale factor using layer_gsd/target_gsd so that it is always <= 1.
    scale_factor = layer_gsd / target_gsd

    # Build the list of parameters. Note the addition of the SCALEFACTOR parameter.
    params = [
        ("VERSION", "2.0.1"),
        ("SERVICE", "wcs"),
        ("REQUEST", "GetCoverage"),
        ("COVERAGEID", layer_name),
        ("FORMAT", "image/tiff"),
        ("SUBSET", f"x({tile_min_x},{tile_max_x})"),
        ("SUBSET", f"y({tile_min_y},{tile_max_y})"),
        ("SCALEFACTOR", f"{scale_factor}"),
    ]

    try:
        response = requests.get(wcs_url, params=params, timeout=10)
    except Exception as e:
        print(f"Exception fetching tile {tile_index}: {e}")
        return None

    if response.status_code == 200:
        try:
            # Open the returned TIFF image and convert/save as PNG.
            # This automatically selects the first 3 bands
            image = Image.open(BytesIO(response.content))
            image.save(output_path)
            print(f"Saved: {output_path}")
            return file_name
        except Exception as e:
            print(f"Error processing image for tile {tile_index}: {e}")
            return None
    else:
        print(f"Error fetching tile {tile_index}: HTTP {response.status_code}")
        return None


def get_min_max_coordinates(shapefile_path):
    """Reads the min and max coordinates from a shapefile."""
    with shapefile.Reader(shapefile_path, encoding="ISO8859-1") as sf:
        min_x, min_y = float("inf"), float("inf")
        max_x, max_y = float("-inf"), float("-inf")
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


def filter_tiles_with_rtree(
    tiles, shapefile_path, tile_width, tile_height, img_width, img_height
):
    """Filters tiles that FULLY contain at least one feature using an R-Tree index."""
    spatial_index = index.Index()
    shapes = []
    out_list = []
    match_count = 0

    # Conversion factors from coordinate to pixel space for a tile
    to_pixel_width = img_width / tile_width
    to_pixel_height = img_height / tile_height

    with shapefile.Reader(shapefile_path, encoding="ISO8859-1") as sf:
        shapes = sf.shapes()
        records = sf.records()
        for i, shape in enumerate(shapes):
            if shape.shapeTypeName == "POLYGON":
                x_coords, y_coords = zip(*shape.points)
                spatial_index.insert(
                    i, (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
                )

    for tile in tiles:
        (tile_min_x, tile_min_y), (tile_max_x, tile_max_y) = tile
        possible_matches = list(
            spatial_index.intersection((tile_min_x, tile_min_y, tile_max_x, tile_max_y))
        )

        if possible_matches:
            matches = []
            for match in possible_matches:
                matched_shape = shapes[match]
                matched_record = records[match]
                x_coords, y_coords = zip(*matched_shape.points)

                # Transform shape to pixel coordinates
                x_coords_pixel = [(x - tile_min_x) * to_pixel_width for x in x_coords]
                y_coords_pixel = [
                    (y - tile_min_y) * -to_pixel_height + img_height for y in y_coords
                ]

                shape_points = list(zip(x_coords_pixel, y_coords_pixel))
                shape_centroid = Polygon(shape_points).centroid
                matches.append(
                    {
                        "points": shape_points,
                        "centroid": (shape_centroid.x, shape_centroid.y),
                        "class": matched_record["dachform"],
                    }
                )
                match_count += 1

            out_list.append({"tile": tile, "contained_polygons": matches})
    print(
        f"Found {match_count} roof polygons that are fully contained in {len(out_list)} tiles."
    )
    return out_list


def process_tile(
    index,
    tile_and_shapes,
    dop_output_dir,
    wcs_url,
    layer_name,
    img_width,
    img_height,
    target_gsd,
    layer_gsd,
    debug,
    debug_data,
):
    """Fetches a DOP image for the tile and updates the dictionary."""
    tile = tile_and_shapes["tile"]
    dop_name = fetch_and_save_dop(
        tile,
        index,
        dop_output_dir,
        wcs_url,
        layer_name,
        img_width,
        img_height,
        target_gsd,
        layer_gsd,
    )
    tile_and_shapes["dop_name"] = dop_name

    if debug:
        debug_data.append((dop_name, tile_and_shapes["contained_polygons"]))
    return dop_name


def plot_tile_and_shapes(file_name, polygons, dop_output_dir, img_width, img_height):
    """
    Plots a tile image along with the overlaid polygons representing buildings.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    file_path = os.path.join(dop_output_dir, file_name)
    if not os.path.exists(file_path):
        print(f"Image file {file_path} not found.")
        return

    image = Image.open(file_path)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, extent=[0, img_width, img_height, 0])

    for polygon in polygons:
        points = polygon["points"]
        polygon_patch = patches.Polygon(
            points, closed=True, edgecolor="red", facecolor="none", linewidth=2
        )
        ax.add_patch(polygon_patch)
        centroid_x, centroid_y = polygon["centroid"]
        ax.plot(centroid_x, centroid_y, "bo", markersize=5, label="Centroid")

    plt.axis("off")
    plt.show()


def delete_path(path):
    """Deletes a file or folder (including subfolders) if the path exists."""
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


def cleanup(paths):
    for path in paths:
        delete_path(path)


def main():
    args = parse_args()
    global DEBUG
    DEBUG = args.debug

    # Setup directories based on output-dir
    OUTPUT_DIR = args.output_dir
    DOP_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "dop_images")
    os.makedirs(DOP_OUTPUT_DIR, exist_ok=True)

    GRU_DOWNLOAD_PATH = os.path.join(OUTPUT_DIR, "gru.zip")
    GRU_EXTRACT_PATH = os.path.join(OUTPUT_DIR, "gru")
    OGR_OUTPUT = os.path.join(OUTPUT_DIR, "roof_shape")
    os.makedirs(OGR_OUTPUT, exist_ok=True)

    # Download and process GRU data
    download_gru_zip(args.gru_url, GRU_DOWNLOAD_PATH)
    # The search string "31001" is used as key for "AX_Gebaeude"
    file_found = extract_relevant_file(GRU_DOWNLOAD_PATH, GRU_EXTRACT_PATH, "31001")
    ogr_output_shp = os.path.join(OGR_OUTPUT, "output_converted.shp")
    if file_found:
        print(f"File containing 'AX_Gebaeude' extracted: {file_found}")
        print("Extracting roof shapes and converting to ESRI shapefile")
        convert_gru_to_shapefile(
            file_found, ogr_output_shp, args.ogr_layer, args.ogr_srs, args.ogr_where
        )
    else:
        print("No file containing 'AX_Gebaeude' was found.")

    # Get bounding coordinates from the shapefile
    min_x, min_y, max_x, max_y = get_min_max_coordinates(ogr_output_shp)
    # Calculate tile dimensions in map units using target GSD and image size.
    tile_width = args.img_width * args.target_gsd
    tile_height = args.img_height * args.target_gsd
    tiles = generate_tiles(min_x, min_y, max_x, max_y, tile_width, tile_height)
    filtered_shapes = filter_tiles_with_rtree(
        tiles, ogr_output_shp, tile_width, tile_height, args.img_width, args.img_height
    )

    debug_data = []  # Collect debug data if needed

    # Process tiles in parallel using a ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.wcs_workers) as executor:
        futures = {
            executor.submit(
                process_tile,
                i,
                tile_dict,
                DOP_OUTPUT_DIR,
                args.wcs_url,
                args.layer_name,
                args.img_width,
                args.img_height,
                args.target_gsd,
                args.layer_gsd,
                DEBUG,
                debug_data,
            ): tile_dict
            for i, tile_dict in enumerate(filtered_shapes)
        }

        for future in tqdm(as_completed(futures), total=len(filtered_shapes)):
            dop_name = future.result()

    # Save metadata as JSON
    metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as fp:
        json.dump(filtered_shapes, fp)
    print(f"Metadata saved to {metadata_path}")

    # Plot debug images sequentially if DEBUG is enabled
    if DEBUG:
        for dop_name, polygons in debug_data:
            plot_tile_and_shapes(
                dop_name, polygons, DOP_OUTPUT_DIR, args.img_width, args.img_height
            )

    # Cleanup temporary files
    cleanup([GRU_DOWNLOAD_PATH, GRU_EXTRACT_PATH, OGR_OUTPUT])


if __name__ == "__main__":
    main()
