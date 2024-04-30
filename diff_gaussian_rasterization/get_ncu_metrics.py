import sys

def extract_metrics_from_log(log_filename):
    total_dram_bytes = 0.0
    total_gpu_time = 0.0
    
    with open(log_filename, 'r') as file:
        for line in file:
            if "dram__bytes.sum" in line:
                parts = line.split()
                dram_bytes = float(parts[-1])
                if "Mbyte" in line:
                    dram_bytes *= 1024  # Convert MB to KB
                total_dram_bytes += dram_bytes
            elif "gpu__time_duration.sum" in line:
                parts = line.split()
                gpu_time = float(parts[-1])
                total_gpu_time += gpu_time

    return total_dram_bytes, total_gpu_time

# Main function to handle command line arguments
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_log_file>")
        sys.exit(1)

    log_filename = sys.argv[1]
    total_dram_bytes, total_gpu_time = extract_metrics_from_log(log_filename)
    print(f"Total DRAM Bytes: {total_dram_bytes} KB, {total_dram_bytes/1024} MB")
    print(f"Total GPU Time: {total_gpu_time} usecond")
