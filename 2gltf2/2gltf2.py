# 
# The MIT License (MIT)
#
# Copyright (c) since 2017 UX3D GmbH
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import re

#
# Imports
#

import bpy
from mathutils import Vector
import os
import sys

#
# Globals
#

#
# Functions
#
# Function to load an image into Blender, given a filepath
def load_image(image_path):
    if os.path.exists(image_path):
        return bpy.data.images.load(image_path)
    else:
        print(f"Image not found: {image_path}")
        return None

def main():
    # read two arguments: the input file path and the output basename
    if len(sys.argv) < 3:
        print("Usage: blender -b -P 2gltf2.py -- <input_file> <output_basename>")
        return
    # find the -- argument
    input_path = sys.argv[sys.argv.index("--") + 1]
    output_basename = sys.argv[sys.argv.index("--") + 2]
    root, current_extension = os.path.splitext(input_path)
    current_directory = os.path.dirname(input_path)

    if current_extension != ".abc" and current_extension != ".blend" and current_extension != ".dae" and current_extension != ".fbx" and current_extension != ".obj" and current_extension != ".ply" and current_extension != ".stl" and current_extension != ".usd" and current_extension != ".usda" and current_extension != ".usdc" and current_extension != ".wrl" and current_extension != ".x3d":
        print("Unsupported file format: " + current_extension)
        return

    bpy.ops.wm.read_factory_settings(use_empty=True)
    print("Converting: '" + input_path + "'")

    if current_extension == ".abc":
        bpy.ops.wm.alembic_import(filepath=input_path)

    if current_extension == ".blend":
        bpy.ops.wm.open_mainfile(filepath=input_path)

    if current_extension == ".dae":
        bpy.ops.wm.collada_import(filepath=input_path)

    if current_extension == ".fbx":
        bpy.ops.import_scene.fbx(filepath=input_path)

    if current_extension == ".obj":
        bpy.ops.import_scene.obj(filepath=input_path)

    if current_extension == ".ply":
        bpy.ops.import_mesh.ply(filepath=input_path)

    if current_extension == ".stl":
        bpy.ops.import_mesh.stl(filepath=input_path)

    if current_extension == ".usd" or current_extension == ".usda" or current_extension == ".usdc":
        bpy.ops.wm.usd_import(filepath=input_path)

    if current_extension == ".wrl" or current_extension == ".x3d":
        bpy.ops.import_scene.x3d(filepath=input_path)

    # import roughness
    # Function to find the roughness texture based on the material name
    def find_roughness_texture(material_name):
        # Extract the numeric part of the material name (e.g., "mat_shape0_0" -> "00")
        match = re.search(r'(\d+)', material_name)
        if match:
            # Assuming the material name includes a number that matches the texture file name
            number = match.group(1).zfill(2)  # Ensure it has at least two digits
            roughness_filename = f"{number}_roughness.png"
            return os.path.join(current_directory, roughness_filename)
        else:
            print(f"No numeric part found in material name: {material_name}")
            return None


    # Iterate through all materials
    for mat in bpy.data.materials:
        roughness_path = find_roughness_texture(mat.name)
        if roughness_path and os.path.exists(roughness_path):
            if not mat.use_nodes:
                mat.use_nodes = True
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            bsdf_node = None

            # Find the Principled BSDF node
            for node in nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    bsdf_node = node
                    break

            # If Principled BSDF node doesn't exist, create one
            if not bsdf_node:
                bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')
                bsdf_node.location = (0, 0)

            # Check if the Roughness texture is already connected
            roughness_connected = False
            for link in links:
                if link.to_node == bsdf_node and link.to_socket.name == 'Roughness':
                    roughness_connected = True
                    break

            # If Roughness texture is not connected, load and connect it
            if not roughness_connected:
                print(f"linking roughness: {roughness_path} to {mat.name}")
                roughness_image = load_image(roughness_path)
                if roughness_image:
                    roughness_node = nodes.new(type='ShaderNodeTexImage')
                    roughness_node.image = roughness_image
                    roughness_node.location = bsdf_node.location + Vector((-300, 0))
                    links.new(roughness_node.outputs['Color'], bsdf_node.inputs['Roughness'])

    export_file = current_directory + "/" + output_basename + ".gltf"
    print("Writing: '" + export_file + "'")
    bpy.ops.export_scene.gltf(filepath=export_file)

if __name__ == "__main__":
    main()