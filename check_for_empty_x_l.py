import json
import os

def filter_jsonl(input_file: str, output_file: str):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            try:
                data = json.loads(line)
                # Check if "x_l" exists and has 3 or more words
                if "x_l" in data and len(data["x_l"].split()) >= 3:
                    outfile.write(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line in {input_file}.")

def process_files(file_list):
    for file_path in file_list:
        output_file = f"{os.path.splitext(file_path)[0]}.filtered.jsonl"
        filter_jsonl(file_path, output_file)
        print(f"Processed {file_path} -> {output_file}")

# List of files to process
files_to_process = [
    'gen_data/news.orig.part_00.shuf.jsonl',
    'gen_data/news.orig.part_01.shuf.jsonl',
    'gen_data/news.orig.part_02.shuf.jsonl',
    'gen_data/news.orig.part_03.shuf.jsonl',
    'gen_data/news.orig.part_04.shuf.jsonl',
    'gen_data/news.orig.part_05.shuf.jsonl'
]

# Process all files
process_files(files_to_process)
