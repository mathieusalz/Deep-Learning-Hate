import re

# Input and output file paths
input_file = "slurm-2645381.out"
output_file = "filteredTrials_perLang.txt"

# Regex pattern for lines like:
# [I 2025-05-20 10:54:34,264] Trial 0 finished with value: ...
pattern = re.compile(r'^\[I .*?Trial \d+ finished with value:')

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        if pattern.match(line):
            outfile.write(line)
