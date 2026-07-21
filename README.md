# Smile
Utility to automatically stitch sequences of intraoral images from a low cost dental camera into a synthetic panoramic view.

Author: [Mark Hsieh](https://github.com/mcmhsieh)

## Getting Started (Microsoft Windows)

### Setting up
- Clone https://github.com/mcmhsieh/Smile.git or download a copy of the repository
- If your system supports CUDA, install an up-to-date NVIDIA Graphics Driver
  - *(there should be no need to separately install the CUDA runtime because it is already bundled with PyTorch)*
- Install Python 3.11 and Python Poetry
- Create a virtual environment, activate it, and use Poetry to install the dependencies
  - *(the installed dependencies should occupy ~6-7GB of disk space)*

### Initial test run on a very short sequence
Activate the virtual environment and run the stages of the pipeline in sequence from the cloned repository's `smile` subdirectory:

    cd smile
    python.exe calc_sequential_flow_and_blur.py
    python.exe select_key_frames.py
    python.exe stitch_key_frames.py
    python.exe select_and_position_inter_key_aux_frames.py
    python.exe compute_depth_images.py
    python.exe integrate_depth_images.py
    python.exe view_synthesis.py

If everything is installed and working correctly, the example JPEG images included in the cloned repository's `pipeline-input\example-30frames-iso46to46` subdirectory should be stitched together to generate a synthetic view saved as a JPEG image (with a timestamped filename) in the `pipeline-workspace/example-30frames-iso46to46/view_synthesis` subdirectory.  
![Example synthesised view](docs/images/example-30frames-iso46to46-view_synthesis-output.jpg)
