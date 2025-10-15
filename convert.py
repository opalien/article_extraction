import os
from typing import List, Tuple, Optional

def process_file(file_path: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Reads a file and returns its content. Handles potential UnicodeDecodeError.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as input_file:
            content = input_file.read()
        return file_path, content, None
    except UnicodeDecodeError:
        return file_path, None, "Failed to decode the file, as it is not saved with UTF-8 encoding."

def write_directory_structure_to_file(directory_path: str, output_file_name: str, exclude: List[str]) -> Tuple[int, int]:
    copied_files = 0

    file_list: List[str] = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            if not any(ex in file_path for ex in exclude):
                file_list.append(file_path)
    
    total_files = len(file_list)

    with open(output_file_name, 'w', encoding='utf-8') as output_file:
        for i, file_path in enumerate(file_list):
            print(f"Processing file {i + 1}/{total_files}: {file_path}")
            path, content, error_message = process_file(file_path)
            
            if content is not None:
                file_name_line = f"--- START FILE: {path} ---\n"
                output_file.write(file_name_line)
                output_file.write(content + "\n")
                end_line = f"--- END FILE: {path} ---\n"
                output_file.write(end_line)
                copied_files += 1
            
            if error_message is not None:
                output_file.write(f"{path}\n{error_message}\n")

    return total_files, copied_files

if __name__ == "__main__":

    input_directory = "."
    output_file_name = "project" + ".txt"

    exclude = [
        "converter.py", 
        "project.txt", 

        ".git/", 
        ".nix", 
        ".env",
        ".csv",
        ".json",
        ".png",
        ".pyc",
        ".zip",
        ".db",
        ".cache/"
        
    ]

    total_files, copied_files = write_directory_structure_to_file(input_directory, output_file_name, exclude)

    print(f"\nThere are a total of {total_files} files to process in the '{input_directory}' directory (after excluding files).")
    print(f"A total of {copied_files} files have been copied to '{output_file_name}'.")