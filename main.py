import subprocess
import os
import threading

def stream_reader(stream, buffer):
    while True:
        line = stream.readline()
        if line:
            print(line.strip())
            buffer.append(line)
        else:
            break

# Get parameters from user input
script_path = input("Enter the path to the detect.py script (e.g., yolov5/detect.py): ")
source = input("Enter the video source file path: ")
view_img = input("Do you want to view the image? (yes/no): ").lower() == 'yes'
conf_thresh = float(input("Enter the confidence threshold (e.g., 0.1): "))
device = int(input("Enter the device ID (e.g., 0 for GPU): "))
weights = input("Enter the weights file path (e.g., yolov5s.pt): ")
half = input("Use half precision? (yes/no): ").lower() == 'yes'

# Construct the command as a list of arguments
command = [
    "python", script_path,
    "--source", source,
    "--conf-thres", str(conf_thresh),
    "--device", str(device),
    "--weights", weights
]
if view_img:
    command.append("--view-img")
if half:
    command.append("--half")

# Print the constructed command (for debugging)
print(f"Constructed command: {' '.join(command)}")

# Set the working directory to the directory containing detect.py
working_directory = os.path.dirname(script_path)

# Run the command with real-time output
process = subprocess.Popen(command, cwd=working_directory, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

# Create buffers to store the output
stdout_buffer = []
stderr_buffer = []

# Create and start threads to read stdout and stderr
stdout_thread = threading.Thread(target=stream_reader, args=(process.stdout, stdout_buffer))
stderr_thread = threading.Thread(target=stream_reader, args=(process.stderr, stderr_buffer))
stdout_thread.start()
stderr_thread.start()

# Wait for the process to complete and the threads to finish
process.wait()
stdout_thread.join()
stderr_thread.join()

# Print any remaining output (if needed)
print("Remaining output:\n", ''.join(stdout_buffer))
if stderr_buffer:
    print("Remaining errors:\n", ''.join(stderr_buffer))
