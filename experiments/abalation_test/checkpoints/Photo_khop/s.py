# Data from the table
data = [
            {"k_hops": 1, "mean": 0.9092, "std": 0.0106},
            {"k_hops": 2, "mean": 0.9086, "std": 0.0021},
            {"k_hops": 4, "mean": 0.8796, "std": 0.0109},
            {"k_hops": 5, "mean": 0.8736, "std": 0.0076},
            {"k_hops": 6, "mean": 0.8784, "std": 0.0140},
            {"k_hops": 7, "mean": 0.8832, "std": 0.0030},
            {"k_hops": 8, "mean": 0.8774, "std": 0.0165},
            {"k_hops": 9, "mean": 0.8830, "std": 0.0076},
            {"k_hops": 10, "mean": 0.8772, "std": 0.0081},
]

# Generate .txt files
file_paths = []
for entry in data:
    file_name = f"MultiHop_GRAND_K_hops_{entry['k_hops']}.txt"
    content = f"mean: {entry['mean']}, std: {entry['std']}"
    file_path = f"{file_name}"
    file_paths.append(file_path)
    with open(file_path, "w") as file:
        file.write(content)

file_paths
