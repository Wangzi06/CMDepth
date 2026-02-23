import os
import sys
from pathlib import Path

def remove_raw_data_prefix(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            parts = line.strip().split()
            if len(parts) >= 2:
                # Remove '/raw_data' from the first path
                if parts[0].startswith('/raw_data'):
                    parts[0] = parts[0][9:]  # Remove first 9 characters ('/raw_data')
                
                # Join the parts back together
                modified_line = ' '.join(parts)
                f_out.write(modified_line + '\n')
            else:
                # Write unchanged if format is unexpected
                f_out.write(line)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python remove_prefix.py input_file output_file")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    remove_raw_data_prefix(input_file, output_file)
    print(f"Processing complete. Modified data saved to {output_file}")