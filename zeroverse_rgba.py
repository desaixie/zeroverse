import sys
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import argparse
import bpy
from mathutils import Vector
import pickle
import random
import numpy as np
import json
import cv2
import concurrent.futures
import time
import os
from rich import print
import tempfile
import copy
from multiprocessing import cpu_count
import tempfile
import uuid
import augment_shape


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--object_path",
    type=str,
    required=True,
    help="Path to the object file",
)
parser.add_argument("--output_dir", type=str, default="../renderings")
parser.add_argument("--ibl_path", type=str, default="")
parser.add_argument(
    "--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"]
)
parser.add_argument("--save_norm_glb", action="store_true", help="Save normalized glb")
parser.add_argument("--only_use_cpu", action="store_true", help="Use CPU rendering")
parser.add_argument(
    "--keep_exr", action="store_true", help="Keep EXR files after rendering"
)
parser.add_argument(
    "--no_tonemap", action="store_true", help="Do not tonemap the images"
)
parser.add_argument("--local_cache_dir", type=str, default="../local_cache")
parser.add_argument("--boolean_probability", type=float, default=0.0)
parser.add_argument("--wireframe_probability", type=float, default=0.0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--radius_min', type=float, default=1.5)
parser.add_argument('--radius_max', type=float, default=2.8)
args = parser.parse_args()

raw_args = copy.deepcopy(args)


# Set up temp dir
print(f"old temp_dir: {tempfile.gettempdir()}")
temp_dir = os.path.join(args.local_cache_dir, "tmp", str(uuid.uuid4()))
os.makedirs(temp_dir, exist_ok=True)
tempfile.tempdir = temp_dir
print(f"new temp_dir: {tempfile.gettempdir()}")

# Detect devices
bpy.context.preferences.addons["cycles"].preferences.get_devices()

# Use OptiX
bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "OPTIX"
bpy.context.scene.cycles.device = "GPU"
bpy.context.preferences.addons["cycles"].preferences.get_devices()

for d in bpy.context.preferences.addons["cycles"].preferences.devices:
    d["use"] = 1  # Using all devices, include GPU and CPU


if args.only_use_cpu:
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "NONE"
    bpy.context.scene.cycles.device = 'CPU'

    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        if d["type"] == "GPU":
            d["use"] = 0  # disable GPU

print(
    f"bpy.context.preferences.addons['cycles'].preferences.compute_device_type: {bpy.context.preferences.addons['cycles'].preferences.compute_device_type}"
)

# Speed up rendering of the same scene
bpy.context.scene.render.use_persistent_data = True

context = bpy.context
scene = context.scene
render = scene.render

render.engine = args.engine
render.image_settings.file_format = "OPEN_EXR"
render.image_settings.color_mode = "RGBA"
render.resolution_x = 512
render.resolution_y = 512
render.resolution_percentage = 100

scene.cycles.samples = 64
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 1.5
scene.cycles.use_denoising = True
scene.render.film_transparent = True


def add_uniform_lighting():
    # Add uniform lighting
    bpy.context.scene.world = bpy.data.worlds.new("UniformWorld")
    bpy.context.scene.world.use_nodes = True
    shader = bpy.context.scene.world.node_tree.nodes["Background"]
    shader.inputs[0].default_value = (1, 1, 1, 1)  # RGB + Alpha
    shader.inputs[1].default_value = 1.0  # Strength


def add_envmap_lighting(filepath):
    # Load image
    img = bpy.data.images.load(filepath)

    # Create new world material
    world = bpy.data.worlds.get("EnvmapWorld")
    if world is None:
        world = bpy.data.worlds.new("EnvmapWorld")

    world.use_nodes = True

    # Get the node tree
    node_tree = world.node_tree

    # clear all nodes to start clean
    node_tree.nodes.clear()

    # create new environment texture node and set the image
    env_tex_node = node_tree.nodes.new(type="ShaderNodeTexEnvironment")
    env_tex_node.image = img
    env_tex_node.location = (-300, 300)

    # create new background node
    bg_node = node_tree.nodes.new(type="ShaderNodeBackground")
    bg_node.location = (100, 300)

    # create new output node
    out_node = node_tree.nodes.new(type="ShaderNodeOutputWorld")
    out_node.location = (300, 300)

    # link nodes together
    node_tree.links.new(env_tex_node.outputs["Color"], bg_node.inputs["Color"])
    node_tree.links.new(bg_node.outputs["Background"], out_node.inputs["Surface"])

    # set the new world as the active world
    bpy.context.scene.world = world


def reset_scene():
    """Resets the scene to a clean state."""
    # delete everything: object, camera, light
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def load_object(object_path):
    """Loads a glb model into the scene."""
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")


def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (np.inf,) * 3
    bbox_max = (-np.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return np.array(bbox_min), np.array(bbox_max)


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def normalize_scene():
    bbox_min, bbox_max = scene_bbox()
    scale = 1.8 / np.max(bbox_max - bbox_min)
    offset = -(bbox_min + bbox_max) / 2
    offset = Vector((offset[0], offset[1], offset[2]))
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
        obj.matrix_world.translation *= scale
        obj.scale *= scale
    bpy.context.view_layer.update()
    bpy.ops.object.select_all(action="DESELECT")


def add_camera():
    bpy.ops.object.camera_add(location=(0.0, 0.0, 0.0))
    camera_object = bpy.context.object
    scene.camera = camera_object  # make this the current camera

    camera_object.location = (0, 1.2, 0)
    hfov = 50
    camera_object.data.sensor_width = 32
    camera_object.data.lens = camera_object.data.sensor_width / (
        2 * np.tan(np.deg2rad(hfov / 2))
    )
    cam_constraint = camera_object.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    # create an empty object to track; look at (0, 0, 0)
    empty = bpy.data.objects.new("Empty", None)
    empty.location = (0, 0, 0)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty
    return camera_object


def turntable_sample_cam_loc(num_samples=4, radius=3.0):
    theta = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
    phi = np.deg2rad(np.ones_like(theta) * 20)
    cam_locations = np.stack(
        [
            radius * np.cos(phi) * np.cos(theta),
            radius * np.cos(phi) * np.sin(theta),
            radius * np.sin(phi),
        ],
        axis=1,
    )
    return cam_locations


# source Zero 1-to-3: https://github.com/cvlab-columbia/zero123/blob/main/objaverse-rendering/scripts/blender_script.py
def sample_cam_loc(
    num_samples=32, radius_min=1.5, radius_max=2.8, maxz=1.6, minz=-0.75
):
    cam_locations = []
    for r in range(num_samples):
        correct = False
        while not correct:
            vec = np.random.uniform(-1, 1, 3)
            radius = np.random.uniform(radius_min, radius_max, 1)
            vec = vec / np.linalg.norm(vec, axis=0) * radius[0]
            if maxz > vec[2] > minz:
                correct = True
        cam_locations.append(vec)
    return np.array(cam_locations)


def jiahao_sample_cam_loc(num_samples=16, radius=2.7):
    theta = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
    phi = np.deg2rad(np.ones_like(theta) * 20)
    cam_locations = np.stack(
        [
            radius * np.cos(phi) * np.cos(theta),
            radius * np.cos(phi) * np.sin(theta),
            radius * np.sin(phi),
        ],
        axis=1,
    )
    return cam_locations


def gso_sample_cam_loc(num_samples=64, radius=2.7, elevations=[0, 20, 40, 60]):
    num_elevations = len(elevations)
    num_theta = num_samples // num_elevations
    theta = np.tile(np.linspace(0, 2 * np.pi, num_theta, endpoint=False), num_elevations)
    phi = np.deg2rad(np.repeat(elevations, num_theta))
    cam_locations = np.stack(
        [
            radius * np.cos(phi) * np.cos(theta),
            radius * np.cos(phi) * np.sin(theta),
            radius * np.sin(phi),
            ],
        axis=1,
    )
    return cam_locations


def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list


def get_camera_params(camera_object):
    c2w = np.array(listify_matrix(camera_object.matrix_world))
    resolution_x = render.resolution_x
    resolution_y = render.resolution_y
    cx = resolution_x / 2.0
    cy = resolution_y / 2.0
    fx = cx / (camera_object.data.sensor_width / 2.0 / camera_object.data.lens)
    fy = fx
    w2c = np.linalg.inv(c2w)
    w2c = np.diag([1.0, -1.0, -1.0, 1.0]) @ w2c

    cam_dict = {
        "w": resolution_x,
        "h": resolution_y,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "w2c": w2c.tolist(),
    }
    return cam_dict


def read_one_image(fpath):
    im = cv2.imread(fpath, -1)
    im, alpha = im[:, :, :3], im[:, :, 3]
    valid_pixels = im[alpha > 0.95]
    minval, maxval = np.percentile(valid_pixels, [1, 99])
    return (im, alpha, minval, maxval, fpath)


def read_images_parallel(fpaths):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        threads = [executor.submit(read_one_image, f) for f in fpaths]
        return [t.result() for t in threads]


def write_one_image(im_data, use_white_bg=True):
    im, alpha, minval, maxval, fpath = im_data

    valid_mask = alpha > 1e-3
    if np.any(valid_mask):
        valid_pixels = im[valid_mask] / alpha[valid_mask][:, None]
        valid_pixels = (valid_pixels - minval) / (maxval - minval)
        im[valid_mask] = valid_pixels
    im = np.clip(im, 0.0, 1.0)
    # blender by default uses black background; we use white background
    if use_white_bg:
        im = im * alpha[:, :, None] + np.ones_like(im) * (1.0 - alpha[:, :, None])
    im = np.power(im, 1.0 / 2.2)

    im = (im * 255.0).clip(0.0, 255.0).astype(np.uint8)
    alpha = (alpha * 255.0).clip(0.0, 255.0).astype(np.uint8)

    im = np.concatenate([im, alpha[:, :, None]], axis=2)
    cv2.imwrite(fpath, im)


def write_images_parallel(im_datas):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        threads = [executor.submit(write_one_image, im_data) for im_data in im_datas]
        return [t.result() for t in threads]


def tonemap_folder(rendering_dir, keep_exr=False):
    exr_fpaths = [
        os.path.join(rendering_dir, f)
        for f in os.listdir(rendering_dir)
        if f.endswith("_rgba.exr")
    ]
    im_datas = read_images_parallel(exr_fpaths)
    mean_minval = np.mean([d[2] for d in im_datas])
    mean_maxval = np.mean([d[3] for d in im_datas])
    print(f"Minval: {mean_minval}, maxval: {mean_maxval}")
    with open(os.path.join(rendering_dir, "../minmax.txt"), "w") as f:
        f.write(f"{mean_minval} {mean_maxval}")

    png_fpaths = [f.replace("_rgba.exr", "_rgba.png") for f in exr_fpaths]
    for idx in range(len(png_fpaths)):
        im_datas[idx] = im_datas[idx][:2] + (
            mean_minval,
            mean_maxval,
            png_fpaths[idx],
        )
    write_images_parallel(im_datas)

    if not keep_exr:
        for f in exr_fpaths:
            os.remove(f)


def render_images():
    try:
        """Saves rendered images of the object in the scene."""
        os.makedirs(args.output_dir, exist_ok=True)

        reset_scene()
        obj_name = augment_shape.load_object_return_name(args.object_path)
        normalize_scene()
        # add a boolean operation with a primitive
        if args.boolean_probability > 0: # skip if boolean_probability is set to 0
            augment_parameters = augment_shape.augment_with_boolean(obj_name, cut_type=None, probability=args.boolean_probability)
            if augment_parameters['is_augmented']:
                out_shape_dir = os.path.join(args.output_dir, "shape")
                os.makedirs(f'{out_shape_dir}', exist_ok=True)

                augmented_fn = os.path.join(out_shape_dir, args.object_path.split("/")[-1].replace(".glb", "_augmented.glb"))
                bpy.ops.export_scene.gltf(filepath=augmented_fn)
                json_output_fn = augmented_fn.replace('.glb', '_parameters.json')
                with open(json_output_fn, 'w') as f:
                    json.dump(augment_parameters, f, indent=4, cls=NpEncoder)
                print(f'Saved {json_output_fn}')

        if args.wireframe_probability > 0:
            wireframe_parameters = augment_shape.augment_a_wireframe_primitive(obj_name, random_subdivide_level=True, probability=args.wireframe_probability)
            if wireframe_parameters['is_wireframed']:
                out_shape_dir = os.path.join(args.output_dir, "shape")
                os.makedirs(f'{out_shape_dir}', exist_ok=True)

                augmented_fn = os.path.join(out_shape_dir, args.object_path.split("/")[-1].replace(".glb", "_wireframe.glb"))
                bpy.ops.export_scene.gltf(filepath=augmented_fn)
                json_output_fn = augmented_fn.replace('.glb', '_parameters.json')
                with open(json_output_fn, 'w') as f:
                    json.dump(wireframe_parameters, f, indent=4, cls=NpEncoder)
                print(f'Saved {json_output_fn}')

        seed_everything(args.seed)  # reset seed after augmenting, so the random cameras are seeded separately

        if args.save_norm_glb:
            bpy.ops.export_scene.gltf(
                filepath=os.path.join(args.output_dir, "norm_scene.glb"),
                export_format="GLB",
            )

        if len(args.ibl_path) == 0:
            add_uniform_lighting()
        else:
            add_envmap_lighting(args.ibl_path)
            with open(os.path.join(args.output_dir, "ibl.txt"), "w") as f:
                f.write(args.ibl_path)

        camera = add_camera()

        tic = time.time()
        scene.render.filepath = tempfile.mktemp(suffix=".exr")
        bpy.ops.render.render(write_still=True)
        toc = time.time()
        print(f"Loading rendering kernel takes {toc - tic:.5f}s", flush=True)

        out_dir = args.output_dir
        if os.path.isdir(out_dir):
            os.system(f"rm -rf {out_dir}")

        out_rendering_dir = os.path.join(out_dir, "renderings")
        os.makedirs(out_rendering_dir, exist_ok=True)

        cam_locations = sample_cam_loc(radius_min=args.radius_min, radius_max=args.radius_max)

        opencv_cameras = {"frames": []}
        for idx in range(cam_locations.shape[0]):
            camera.location = cam_locations[idx]

            rgba_path = os.path.join(out_rendering_dir, f"{idx:08d}_rgba.exr")
            scene.render.filepath = rgba_path
            bpy.ops.render.render(write_still=True)

            cam_dict = get_camera_params(camera)
            cam_dict["file_path"] = os.path.relpath(rgba_path, out_dir)
            cam_dict["blender_camera_location"] = cam_locations[idx].tolist()
            opencv_cameras["frames"].append(cam_dict)

        if not args.no_tonemap:
            tonemap_folder(out_rendering_dir, keep_exr=args.keep_exr)
            # change file_path to png
            for frame in opencv_cameras["frames"]:
                frame["file_path"] = frame["file_path"][:-4] + ".png"

        camera_fpath = f"{out_dir}/opencv_cameras.json"
        with open(camera_fpath, "w") as f:
            json.dump(opencv_cameras, f, indent=4)

        # remove temp_dir
        print(f"Removing {temp_dir}")
        os.system(f"rm -rf {temp_dir}")


    except Exception as e:
        # remove temp_dir
        print(f"Removing {temp_dir}")
        os.system(f"rm -rf {temp_dir}")

        print(f"Removing {args.local_cache_dir}")
        os.system(f"rm -rf {args.local_cache_dir}")
        raise e


def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    print(f'Seed: {seed}')


if __name__ == "__main__":
    start_time = time.time()
    seed_everything(args.seed)
    render_images()
    print(f'TIME - zeroverse_rgba.py: rendering time: {time.time() - start_time:.2f}s')
