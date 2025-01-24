# Updated data for HoGA-GAT model with k_hops [1, 2, 10]
updated_hoga_gat_data = [
    {"k_hops": 1, "mean": 0.5294, "std": 0.0392},
    {"k_hops": 2, "mean": 0.6039, "std": 0.0288},
    {"k_hops": 10, "mean": 0.6471, "std": 0.0277},
]

# Generate .txt files for the updated data
updated_hoga_gat_file_paths = []
for entry in updated_hoga_gat_data:
    file_name = f"MultiHop_GAT_K_hops_{entry['k_hops']}.txt"
    content = f"mean: {entry['mean']}, std: {entry['std']}"
    file_path = f"{file_name}"
    updated_hoga_gat_file_paths.append(file_path)
    with open(file_path, "w") as file:
        file.write(content)

updated_hoga_gat_file_paths
