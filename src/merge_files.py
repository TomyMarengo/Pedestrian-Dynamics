# Function to read a txt file into a list of lists
def read_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        return [[float(x) for x in line.split()] for line in lines]

# Function to write the merged list to a new txt file
def write_file(data, filename):
    with open(filename, 'w') as f:
        for line in data:
            f.write('\t'.join(map(str, line)) + '\n')

# Read files
data1 = read_file('../txt/trajectories1.txt')
data2 = read_file('../txt/trajectories2.txt')

# Update ID for the second list
for row in data2:
    row[3] += 14   

# Merge and sort data
merged_data = sorted(data1 + data2, key=lambda x: (x[0], x[3]))

# Filter out rows with Frame >= 252
filtered_data = [row for row in merged_data if row[0] < 252]

# Write to a new file
write_file(filtered_data, '../txt/merged_trajectories.txt')