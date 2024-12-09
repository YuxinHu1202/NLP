import os
import json

def convert_to_jsonl(input_folder, output_file, start_label=0):
    """
    Convert text files in the given folder into a JSONL file, where each line is a text item.
    Each line is a paragraph, and a new label is assigned to each paragraph.

    Args:
        input_folder (str): Folder containing the input txt files.
        output_file (str): The path of the output jsonl file.
        start_label (int): Starting label number for classification (default is 0).
    """
    with open(output_file, 'w', encoding='utf-8') as output:
        label = start_label  # Start from the specified label
        
        # Iterate over all files in the directory
        for filename in os.listdir(input_folder):
            file_path = os.path.join(input_folder, filename)
            
            if os.path.isfile(file_path) and file_path.endswith(".txt"):
                # Read the content of the txt file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()  # Read and strip any extra whitespace
                    
                    if content:  # Skip empty files
                        # Split the content into paragraphs (by newline)
                        paragraphs = content.split('\n')
                        
                        for paragraph in paragraphs:
                            paragraph = paragraph.strip()
                            if paragraph:  # Skip empty paragraphs
                                # Create a JSON object for each paragraph
                                data = {
                                    "label": str(label),  # Use the current label
                                    "text": paragraph
                                }
                                # Write the JSON object as a new line in the output file
                                output.write(json.dumps(data, ensure_ascii=False) + '\n')
                                label += 1  # Increment the label for the next paragraph
                
                print(f"Processed {filename}")

if __name__ == "__main__":
    input_folder = 'review_polarity/txt_sentoken/neg'  # Change this to the folder where your .txt files are
    output_file = 'review_polarity/txt_sentoken/neg.jsonl'  # Change this to the desired output file path
    
    convert_to_jsonl(input_folder, output_file)
    print(f"Conversion complete. JSONL file saved to {output_file}")
