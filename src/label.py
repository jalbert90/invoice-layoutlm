import json
from pathlib import Path

# Obtain path object for file
# Open JSON file in rw mode
# Read in as dict
# Add a 'labels' key
# Make every entry '0' except the 6th, which should be 'seller_name'
# Write dict to JSON

ocr_dir = Path('data/ocr')

gen = ocr_dir.glob('*')
test_json_path = next(gen)

print(test_json_path)

with open(test_json_path, 'r') as f:
    data = json.load(f)

data['labels'] = ['0' for token in data['tokens']]

data['labels'][5] = 'seller_name'

print(data)
