import csv

def split_log(input_filename, output_filename, search_str, delimiter):

    with open(input_filename, 'r') as f:
        lines = f.readlines()

    results = []
    for line in lines:
        if search_str in line:
            items = line.strip().split(delimiter)
            results.append(items[1:])

    with open(output_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(results)
