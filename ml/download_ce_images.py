"""
This script downloads Craft Essence images and metadata from the Atlas Academy API.

This script fetches data for all Craft Essences (CEs) from the North American (NA) 
region of Fate/Grand Order and saves the data in a structured format under 
'ml/dataset/ces'. For each CE, it creates a directory named after the CE's 
collection number (ID) and saves the following:
- metadata.json: A JSON file containing all the CE's data from the API.
- Limit Break images: Full artwork for each limit break level.
"""
import requests
import os
import json
from tqdm import tqdm

BASE_URL = "https://api.atlasacademy.io"
CE_LIST_URL = f"{BASE_URL}/export/NA/nice_equip.json"
OUTPUT_DIR = "ml/dataset/ces"

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

def fetch_and_save_ce_data():
    """
    Fetches the list of Craft Essences, and for each one, downloads its metadata and images.
    """
    # Create the main output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Fetch the list of all CEs from the exported JSON file
    try:
        print("Fetching Craft Essence list from full export...")
        ce_list_response = requests.get(CE_LIST_URL)
        ce_list_response.raise_for_status()
        ce_list = ce_list_response.json()
        print(f"Found {len(ce_list)} Craft Essences.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Craft Essence list: {e}")
        return
    except json.JSONDecodeError:
        print("Error decoding Craft Essence list JSON.")
        return

    # Process each Craft Essence with a progress bar
    for ce_data in tqdm(ce_list, desc="Processing Craft Essences"):
        ce_id = ce_data['collectionNo']
        ce_dir = os.path.join(OUTPUT_DIR, str(ce_id))
        os.makedirs(ce_dir, exist_ok=True)

        print(f"\nProcessing CE {ce_id}: {ce_data.get('name', 'Unknown')}")

        # Save metadata
        metadata_path = os.path.join(ce_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(ce_data, f, ensure_ascii=False, indent=4)
        print(f"Saved metadata for CE {ce_id}")

        # Download all available images from extraAssets
        if 'extraAssets' in ce_data:
            assets = ce_data['extraAssets']
            
            # Download CharaGraph images (main card art)
            if 'charaGraph' in assets and 'equip' in assets['charaGraph']:
                for key, url in assets['charaGraph']['equip'].items():
                    filename = f"card_{key}.png"
                    filepath = os.path.join(ce_dir, filename)
                    download_file(url, filepath)

            # Download Faces
            if 'faces' in assets and 'equip' in assets['faces']:
                for key, url in assets['faces']['equip'].items():
                    filename = f"face_{key}.png"
                    filepath = os.path.join(ce_dir, filename)
                    download_file(url, filepath)
            
            # Download Equip Faces (smaller card icons)
            if 'equipFace' in assets and 'equip' in assets['equipFace']:
                for key, url in assets['equipFace']['equip'].items():
                    filename = f"icon_{key}.png"
                    filepath = os.path.join(ce_dir, filename)
                    download_file(url, filepath)

if __name__ == "__main__":
    fetch_and_save_ce_data() 