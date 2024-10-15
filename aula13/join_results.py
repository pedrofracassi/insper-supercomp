# Read all .txt files in output_pi_recursivo and join them in a single file

import os

# List all folders in output_pi_recursivo
folders = os.listdir('output_pi_recursivo')

# List all files in each folder
files = []

for folder in folders:
    folder_files = os.listdir('output_pi_recursivo/' + folder)
    
    for file in folder_files:
        files.append(folder + '/' + file)

# Open the output file
output_file = open('output_pi_recursivo.txt', 'w')

# Iterate over all files
for file in files:
    # Open the file
    f = open('output_pi_recursivo/' + file, 'r')
    # Read the content
    content = f.read()
    # Write the content to the output file
    output_file.write(content)
    # Close the file
    f.close()

# Close the output file
output_file.close()