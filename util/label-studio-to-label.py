import json
import os

def transform_item(item):
    """
    Parses a single item from the original format to the target format.
    """
    # Extract the filename from the S3 path and remove .png
    img_path = item['img']
    filename_full = os.path.basename(img_path).replace('.png', '')
    parts = filename_full.split('_')
    
    # 1. Construct mat_file
    # Extract azimuth value to create the subdir (e.g., 000066 -> 66Azimuth)
    try:
        az_index = parts.index('azimuth')
        az_val = int(parts[az_index + 1])
    except (ValueError, IndexError):
        az_val = 0
        
    subdir = f"{az_val}Azimuth"
    # Remove the last two components (e.g., imagesS2 and i2)
    mat_base_name = "_".join(parts[:-])
    print(parts)
    mat_file = f"{subdir}/{mat_base_name}.mat"
    
    # 2. Extract Angle (second last component)
    angle = parts[-2]
    
    # 3. Calculate cx and cy
    # Using the first keypoint in the list
    kp = item['kp-1'][0]
    orig_w = kp['original_width']
    orig_h = kp['original_height']
    
    # Multiply original dimensions by the percentage (x/100)
    cx = (kp['x'] / 100) * orig_w
    cy = (kp['y'] / 100) * orig_h
    
    return {
        "mat_file": mat_file,
        "angle": angle,
        "cx": f"{cx:.6f}",
        "cy": f"{cy:.7f}"
    }

def process_labels(input_file, output_file):
    """
    Loads the input JSON, transforms all items, and saves to output JSON.
    """
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Transform the list of items
        transformed_data = [transform_item(item) for item in data]
        
        with open(output_file, 'w') as f:
            json.dump(transformed_data, f, indent=4)
            
        print(f"Successfully converted {len(transformed_data)} items to {output_file}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

# Run the process
process_labels('../label_data.json', 'labels2.json')