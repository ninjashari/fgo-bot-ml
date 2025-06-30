"""
This script downloads servant images and metadata from the Atlas Academy API.

This script fetches data for all servants from the North American (NA) region of Fate/Grand Order,
and saves the data in a structured format under 'ml/dataset/servants'. For each servant,
it creates a directory named after the servant's collection number (ID) and saves the
following:
- metadata.json: A JSON file containing all the servant's data from the API.
- Ascension images: Full-body artwork for each ascension level.
- Costume images: Full-body artwork for any costumes.
- Face images: Cropped face images for each ascension level and costume.
"""
import requests
import os
import json
from tqdm import tqdm

BASE_URL = "https://api.atlasacademy.io"
SERVANT_LIST_URL = f"{BASE_URL}/export/NA/nice_servant.json"
SERVANT_DETAIL_BASE_URL = f"{BASE_URL}/nice/NA/servant"
OUTPUT_DIR = "ml/dataset/servants"

def download_file(url, path):
    """
    Downloads a file from a URL and saves it to a given path with a progress bar.

    Args:
        url (str): The URL of the file to download.
        path (str): The local path to save the file to.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        
        with open(path, 'wb') as f, tqdm(
            total=total_size, unit='iB', unit_scale=True, desc=os.path.basename(path), leave=False
        ) as pbar:
            for data in response.iter_content(block_size):
                pbar.update(len(data))
                f.write(data)

    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")

def fetch_and_save_servant_data():
    """
    Fetches the list of servants, and for each servant, downloads their metadata and images.
    """
    # Create the main output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Fetch the list of all servants from the exported JSON file
    try:
        print("Fetching servant list from full export...")
        servant_list_response = requests.get(SERVANT_LIST_URL)
        servant_list_response.raise_for_status()
        servant_list = servant_list_response.json()
        print(f"Found {len(servant_list)} servants.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching servant list: {e}")
        return

    # Process each servant with a progress bar
    for servant_data in tqdm(servant_list, desc="Processing Servants"):
        servant_id = servant_data['collectionNo']
        servant_dir = os.path.join(OUTPUT_DIR, str(servant_id))
        os.makedirs(servant_dir, exist_ok=True)

        # The data is already detailed, no need for a second fetch
        print(f"\nProcessing servant {servant_id}: {servant_data.get('name', 'Unknown')}")

        # Save metadata
        metadata_path = os.path.join(servant_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(servant_data, f, ensure_ascii=False, indent=4)
        print(f"Saved metadata for servant {servant_id}")

        # Download images
        if 'extraAssets' in servant_data:
            assets = servant_data['extraAssets']
            # Download CharaGraph images (ascensions and costumes)
            if 'charaGraph' in assets and 'ascension' in assets['charaGraph']:
                for key, url in assets['charaGraph']['ascension'].items():
                    filename = f"ascension_{key}.png"
                    filepath = os.path.join(servant_dir, filename)
                    download_file(url, filepath)
            if 'charaGraph' in assets and 'costume' in assets['charaGraph']:
                 for key, url in assets['charaGraph']['costume'].items():
                    filename = f"costume_{key}.png"
                    filepath = os.path.join(servant_dir, filename)
                    download_file(url, filepath)
            
            # Download face images (ascensions and costumes)
            if 'faces' in assets and 'ascension' in assets['faces']:
                for key, url in assets['faces']['ascension'].items():
                    filename = f"face_ascension_{key}.png"
                    filepath = os.path.join(servant_dir, filename)
                    download_file(url, filepath)
            if 'faces' in assets and 'costume' in assets['faces']:
                for key, url in assets['faces']['costume'].items():
                    filename = f"face_costume_{key}.png"
                    filepath = os.path.join(servant_dir, filename)
                    download_file(url, filepath)


if __name__ == "__main__":
    fetch_and_save_servant_data() 