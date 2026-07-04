file_path = "data/african_accessions.txt"

with open(file_path, "r", encoding="utf-8") as file:
    line_count = sum(1 for line in file)

print(f"Total number of lines: {line_count}")
