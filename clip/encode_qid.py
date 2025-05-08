import json
import os
from clip_encode_txt import extract_clip_text_features

def process_jsonl_file(jsonl_path):
    # Create clip_text directory if it doesn't exist
    output_dir = "clip_text"
    os.makedirs(output_dir, exist_ok=True)
    
    # Read and process each line in the JSONL file
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                # Parse JSON line
                data = json.loads(line.strip())
                
                # Extract query and qid
                query = data['query']
                qid = data['qid']
                
                # Create output path
                output_path = os.path.join(output_dir, f"qid{qid}.npz")
                
                # Extract and save features
                print(f"Processing qid {qid}: {query[:50]}...")
                extract_clip_text_features(query, output_path)
                
            except json.JSONDecodeError:
                print(f"Error decoding JSON line: {line}")
                continue
            except Exception as e:
                print(f"Error processing entry: {str(e)}")
                continue

if __name__ == "__main__":
    # Path to your JSONL file
    jsonl_path = "ordered_dataset_thefamilyguy.jsonl"
    process_jsonl_file(jsonl_path)
