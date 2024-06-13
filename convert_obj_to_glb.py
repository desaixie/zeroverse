import sys
import argparse
import subprocess
import time
from pathlib import Path


def convert_files_in_directory(directory, extension):
    directory = Path(directory)
    glbs = []
    for file in directory.rglob(f'*.{extension}'):
        # check if a glb file already exists in the same directory as file
        if list(file.parent.rglob('*.glb')):  # empty list is False, non-empty list is True
            print(f"Skipping file: {file} because a .glb file already exists in the same directory.")
            continue
        print(f"Converting file: {file}")
        # Call the Blender conversion script for each file
        basename = file.parent.name  # uuid/object.obj -> uuid
        subprocess.run(["blender", "-b", "-P", "2gltf2/2gltf2.py", "--", file.resolve(), basename])

    return glbs  # glb Paths


def convert_file(path):
    basename = path.parent.parent.name  # split/uuid/shape/object.obj -> uuid
    print(f"Converting file: {path}")
    # Call the Blender conversion script for each file
    subprocess.run(["blender", "-b", "-P", "2gltf2/2gltf2.py", "--", path.resolve(), basename])
    return str(path / f'{basename}.glb')


def convert_obj_to_glb(dir, extension, write_data_list):

    # Run the conversion on the specified directory
    start_time = time.time()
    glbs = convert_files_in_directory(dir, extension)
    print(f'Converted {len(glbs)} files in {dir}.')

    # save glb paths to data_list.txt
    threeD_dataset_name = 'Zeroverse'
    # don't write the data_list by default
    # when this script is parallelized, each on a single subdir. The main process should write the data_list
    if write_data_list:
        data_list_fpath = Path(dir) / f'{threeD_dataset_name}_data_list.txt'
        with open(data_list_fpath, 'w') as f:
            for glb in glbs:
                f.write(f'{glb.resolve()}\n')
        print(f'Saved absolute glb paths to {data_list_fpath}')

    print(f'Total time: {time.time() - start_time:.2f} seconds. ')
    print(f'Average: {(time.time() - start_time) / len(glbs):.2f} seconds per shape. ')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch convert files to glTF 2.0 format using Blender.')
    parser.add_argument('--dir', type=str, required=True,
                        help='The directory containing Shape__0/, Shape__1/, etc., which contain the .obj files.')
    parser.add_argument('--extension', type=str, default='obj',
                        help='The file extension of the files to convert. Default: obj')
    parser.add_argument('--write_data_list', default=False, action='store_true')
    args = parser.parse_args()
    convert_obj_to_glb(args.dir, args.extension, args.write_data_list)
