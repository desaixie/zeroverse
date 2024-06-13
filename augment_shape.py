import os
import bpy
import numpy as np
import random
from mathutils import Vector
from math import radians


def true_by_chance(probability):
    random_number = random.random()
    return random_number < probability


def load_object_return_name(object_path: str) -> str:
    # Capture current objects in the scene to find the newly added object(s)
    before_import = set(obj.name for obj in bpy.context.scene.objects)

    # Import object based on its file type
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}. Only .glb and .fbx files are supported.")

    # Determine the names of the newly imported object(s)
    after_import = set(obj.name for obj in bpy.context.scene.objects)
    new_objects = after_import - before_import

    # Handle the case where multiple objects are imported
    if not new_objects:
        raise RuntimeError("No new objects were added to the scene.")
    elif len(new_objects) > 1:
        print("Multiple objects imported. Returning the name of the first new object.")

    return next(iter(new_objects))  # Returns the name of one of the new objects


def get_bounding_box_dimensions(obj_name):
    obj = bpy.data.objects.get(obj_name)
    # Ensure the object has a valid bounding box
    if not hasattr(obj, 'bound_box'):
        raise ValueError("Object does not have a bounding box.")

    # Get the bounding box corner points in local space
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]

    # Convert the corner points to a NumPy array for easier manipulation
    bbox_corners_np = np.array([np.array(corner) for corner in bbox_corners])

    # Calculate dimensions
    min_point = bbox_corners_np.min(axis=0)
    max_point = bbox_corners_np.max(axis=0)
    dimensions = max_point - min_point

    return dimensions.tolist()  # Convert from NumPy array to list for width, height, depth


def get_bounding_box_coordinates(obj_name):
    obj = bpy.data.objects.get(obj_name)
    local_coords = [Vector(corner) for corner in obj.bound_box]
    world_coords = [obj.matrix_world @ corner for corner in local_coords]

    return local_coords, world_coords


def apply_transformations(obj_name):
    # Deselect all
    bpy.ops.object.select_all(action='DESELECT')
    # Get the object
    obj = bpy.data.objects[obj_name]
    # Select the object
    obj.select_set(True)
    # Make the object the active object
    bpy.context.view_layer.objects.active = obj
    # Apply the transformation
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    # Deselect all
    bpy.ops.object.select_all(action='DESELECT')


def get_random_vertex_coordinate(object_name):
    obj = bpy.data.objects.get(object_name)
    if obj is None:
        print(f"No object found with the name '{object_name}'.")
        return None

    # Ensure the object has mesh data
    if obj.type != 'MESH':
        print(f"The object '{object_name}' is not a mesh and has no vertices.")
        return None

    mesh = obj.data
    if not mesh.vertices:
        print(f"The mesh '{object_name}' has no vertices.")
        return None

    # Choose a random vertex
    random_vertex = random.choice(mesh.vertices)
    return random_vertex.co  # Return the coordinate of the random vertex


def add_sphere_at_location(location, size):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=size, location=location)
    sphere = bpy.context.active_object
    return sphere


def add_primitive_at_location(location, size, prim_type='sphere', random_type=False, random_rotation=True):
    if random_type:
        prim_type = random.choice(['sphere', 'cube', 'cone', 'cylinder', 'torus'])
        # prim_type = random.choice(['sphere', 'cube', 'cone', 'cylinder'])

    if prim_type == 'sphere':
        bpy.ops.mesh.primitive_uv_sphere_add(radius=size, location=location)
    elif prim_type == 'cube':
        bpy.ops.mesh.primitive_cube_add(size=2 * size, location=location)  # Cube size is edge length
    elif prim_type == 'cone':
        bpy.ops.mesh.primitive_cone_add(radius1=size, location=location)
    elif prim_type == 'cylinder':
        bpy.ops.mesh.primitive_cylinder_add(radius=size, location=location)
    elif prim_type == 'torus':
        bpy.ops.mesh.primitive_torus_add(location=location, major_radius=size, minor_radius=size * 0.3)
    else:
        raise ValueError(f"Unsupported primitive type: {prim_type}")

    # The newly added object becomes the active object in the scene.
    obj = bpy.context.active_object

    # Apply random rotation if requested
    rotation_euler = None
    if random_rotation:
        rotation_euler = tuple(
            (random.uniform(0, 360) for _ in range(3)))  # Generating random rotation angles for X, Y, Z
        obj.rotation_euler = [radians(angle) for angle in rotation_euler]  # Convert angles to radians and apply

    return obj.name, rotation_euler, prim_type


def add_boolean_modifier_to_target(target_name, obj_name, operation='DIFFERENCE', solver='FAST'):
    target_obj = bpy.data.objects.get(target_name)
    operand_obj = bpy.data.objects.get(obj_name)

    if not target_obj or not operand_obj:
        print(f"Could not find specified objects: '{target_name}' or '{obj_name}'.")
        return None

    # Adding the Boolean modifier to the target object
    modifier = target_obj.modifiers.new(name=f"{obj_name}_Boolean", type='BOOLEAN')
    modifier.object = operand_obj
    modifier.operation = operation
    modifier.solver = solver

    return modifier.name


def apply_modifier(obj_name, modifier_name):
    # Retrieve the object by name
    obj = bpy.data.objects.get(obj_name)
    if not obj:
        print(f"Object '{obj_name}' not found.")
        return False

    # Ensure the object has the specified modifier
    modifier = obj.modifiers.get(modifier_name)
    if not modifier:
        print(f"Modifier '{modifier_name}' not found on object '{obj_name}'.")
        return False

    # Apply the modifier
    try:
        bpy.context.view_layer.objects.active = obj  # Set as the active object
        bpy.ops.object.modifier_apply(modifier=modifier_name)
        return True
    except Exception as e:
        print(f"Failed to apply modifier '{modifier_name}' to '{obj_name}': {e}")
        return False


def remove_object_and_data(obj_name):
    obj = bpy.data.objects.get(obj_name)

    if obj is None:
        print(f"Object '{obj_name}' not found!")
        return

    mesh_data = obj.data

    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.ops.object.delete()

    # Check if mesh data is not used elsewhere
    if mesh_data and mesh_data.users == 0:
        bpy.data.meshes.remove(mesh_data)


def add_solidify_modifier(obj_name, thickness=0.01):
    # Retrieve the object by name
    obj = bpy.data.objects.get(obj_name)
    if not obj:
        print(f"Object '{obj_name}' not found.")
        return None

    # Add the Solidify modifier to the object
    solidify_modifier = obj.modifiers.new(name="Solidify", type='SOLIDIFY')
    solidify_modifier.thickness = thickness
    solidify_modifier.offset = 0.5
    return solidify_modifier.name


def get_all_material_names_in_scene():
    # List to store the names of all materials
    material_names = []
    # Loop through all materials in the data block
    for mat in bpy.data.materials:
        material_names.append(mat.name)
    return material_names


def get_a_random_material_name():
    mat_names = get_all_material_names_in_scene()
    mat_name = random.choice(mat_names)

    return mat_name


def link_material_to_object_material_slot(object_name, material_name, slot_index=0):
    # Get the object by name
    obj = bpy.data.objects.get(object_name)
    if obj is None:
        print(f"Object '{object_name}' not found.")
        return

    # Get the material by name
    mat = bpy.data.materials.get(material_name)
    if mat is None:
        print(f"Material '{material_name}' not found.")
        return

    # Check if the object can have materials
    if not hasattr(obj.data, 'materials'):
        print(f"The object '{object_name}' cannot have materials.")
        return

    # Assign the material to the specified slot, or add a new slot if slot_index is None
    if slot_index is not None and slot_index < len(obj.data.materials):
        obj.data.materials[slot_index] = mat
    else:
        obj.data.materials.append(mat)
        if slot_index is not None:
            print(f"Slot index {slot_index} is out of range. Material added to a new slot.")


def add_subdivision_modifier(obj_name, mod_name=None, overwrite=False, **kwargs):
    mod_settings = {
        'type': 'CATMULL_CLARK',
        'levels': 1,
        'render': 1,
        'optimal_display': False,
        'use_limit_surface': False,
        'quality': 3,
        'uv_smooth': 'PRESERVE_BOUNDARIES', # ('NONE', 'PRESERVE_CORNERS', 'PRESERVE_CORNERS_AND_JUNCTIONS', 'PRESERVE_CORNERS_JUNCTIONS_AND_CONCAVE', 'PRESERVE_BOUNDARIES', 'SMOOTH_ALL')
        'boundary_smooth': 'ALL',  # ('PRESERVE_CORNERS', 'ALL')
        'use_creases': True,
        'use_custom_normals': False
    }

    mod_settings.update(kwargs)

    variable_name_mapping = {
        'type': 'subdivision_type',
        'levels': 'levels',
        'render': 'render_levels',
        'optimal_display': 'show_only_control_edges',
        'use_limit_surface': 'use_limit_surface',
        'quality': 'quality',
        'uv_smooth': 'uv_smooth',
        'boundary_smooth': 'boundary_smooth',
        'use_creases': 'use_creases',
        'use_custom_normals': 'use_custom_normals'
    }

    if mod_name is None:
        mod_name = 'my_subdivision_modifer'

    if obj_name not in bpy.data.objects:
        raise ValueError("Object does not exist in the scene.")

    obj = bpy.data.objects[obj_name]

    # If the specified mod_name exists
    if mod_name in obj.modifiers:
        if overwrite:
            obj.modifiers.remove(obj.modifiers[mod_name])
        else:
            # Create a new unique modifier name
            i = 1
            new_mod_name = f"{mod_name}.{str(i).zfill(3)}"
            while new_mod_name in obj.modifiers:
                i += 1
                new_mod_name = f"{mod_name}.{str(i).zfill(3)}"
            mod_name = new_mod_name

    mod = obj.modifiers.new(name=mod_name, type='SUBSURF')
    for key, value in mod_settings.items():
        setattr(mod, variable_name_mapping[key], value)
    return mod.name, mod_settings


def add_wireframe_modifier(obj_name, mod_name=None, overwrite=False, **kwargs):
    mod_settings = {
        'thickness': 0.0002,
        'offset': 0,
        'boundary': False,
        'replace_original': True,
        'even': True,
        'relatvie': False,
        'crease_edges': False,
        'crease_weight': 1.0,
        'material_offset': 1,
        'vertex_group': '',
        'factor': 0
    }

    mod_settings.update(kwargs)

    variable_name_mapping = {
        'thickness': 'thickness',
        'offset': 'offset',
        'boundary': 'use_boundary',
        'replace_original': 'use_replace',
        'even': 'use_even_offset',
        'relatvie': 'use_relative_offset',
        'crease_edges': 'use_crease',
        'crease_weight': 'crease_weight',
        'material_offset': 'material_offset',
        'vertex_group': 'vertex_group',
        'factor': 'thickness_vertex_group'
    }

    if mod_name is None:
        mod_name = 'my_wireframe_modifer'

    if obj_name not in bpy.data.objects:
        raise ValueError("Object does not exist in the scene.")

    obj = bpy.data.objects[obj_name]

    # If the specified mod_name exists
    if mod_name in obj.modifiers:
        if overwrite:
            obj.modifiers.remove(obj.modifiers[mod_name])
        else:
            # Create a new unique modifier name
            i = 1
            new_mod_name = f"{mod_name}.{str(i).zfill(3)}"
            while new_mod_name in obj.modifiers:
                i += 1
                new_mod_name = f"{mod_name}.{str(i).zfill(3)}"
            mod_name = new_mod_name

    mod = obj.modifiers.new(name=mod_name, type='WIREFRAME')
    vg_name = mod_settings['vertex_group']
    if vg_name != '':
        if vg_name not in obj.vertex_groups:  # ensure the vertex group exists
            raise ValueError(f"Vertex group {vg_name} does not exist in object {obj_name}.")

    for key, value in mod_settings.items():
        setattr(mod, variable_name_mapping[key], value)
    return mod.name, mod_settings


def augment_with_boolean(obj_name, cut_type=None, size=None, thickness=None, probability=1):
    if size == None:
        size = random.uniform(0.3, 0.6)
    if thickness == None:
        thickness = random.uniform(0.01, 0.02)
    if cut_type == None:
        random_type = True
        cut_type = 'sphere'
    else:
        random_type = False

    augment_parameters = {}
    if true_by_chance(probability):
        pos_0 = get_random_vertex_coordinate(obj_name)
        pos_0_list = list(pos_0)

        # cutter = add_primitive_at_location(pos_0, size, random_type=True)
        cutter, rotation_euler, cut_type = add_primitive_at_location(pos_0, size, prim_type=cut_type,
                                                                     random_type=random_type)

        bool_mod_name = add_boolean_modifier_to_target(obj_name, cutter, operation='DIFFERENCE', solver='FAST')
        print(bool_mod_name)
        if bool_mod_name:
            print(f"Added Boolean modifier: {bool_mod_name}")
            apply_modifier(obj_name, bool_mod_name)
            remove_object_and_data(cutter)
            solid_mod_name = add_solidify_modifier(obj_name, thickness)
            if solid_mod_name:
                apply_modifier(obj_name, solid_mod_name)
        augment_parameters['is_augmented'] = True
        augment_parameters['size'] = size
        augment_parameters['pos_0'] = pos_0_list  # cannot pickle Vector object
        augment_parameters['cut_type'] = cut_type
        augment_parameters['rotation_euler'] = rotation_euler
        augment_parameters['thickness'] = thickness
    else:
        augment_parameters['is_augmented'] = False
        print('Skip boolean operation.')
    return augment_parameters


def augment_a_wireframe_primitive(obj_name, cut_type=None, size=None, thickness=None, subdivide_type=None, random_subdivide_type=True, subdivide_level=0, random_subdivide_level=True, probability=1):
    if size is None:
        size = random.uniform(0.3, 0.6)
    if thickness is None:
        thickness = random.uniform(0.01, 0.03)
    if cut_type is None:
        random_type = True
        cut_type = 'sphere'
    else:
        random_type = False

    if subdivide_type is None:
        subdivide_type = 'CATMULL_CLARK'

    if random_subdivide_type:
        subdivide_type = random.choice(['SIMPLE', 'CATMULL_CLARK'])

    if random_subdivide_level:
        subdivide_level = random.choice([0, 1])

    wireframe_paremeters = {}
    if true_by_chance(probability):
        pos_0 = get_random_vertex_coordinate(obj_name)
        pos_0_list = list(pos_0)

        # cutter = add_primitive_at_location(pos_0, size, random_type=True)
        cutter, rotation_euler, cutter_type = add_primitive_at_location(pos_0, size, prim_type=cut_type, random_type=random_type)
        mat_name = get_a_random_material_name()
        link_material_to_object_material_slot(cutter, mat_name)

        if cutter_type == 'cube':
            if subdivide_level > 0:
                subdiv_mod_name, _ = add_subdivision_modifier(cutter, type=subdivide_type, levels=subdivide_level + 2, render=subdivide_level + 2, overwrite=True)
            else:
                subdiv_mod_name = None

        else:
            if subdivide_level > 0:
                subdiv_mod_name, _ = add_subdivision_modifier(cutter, type=subdivide_type, levels=subdivide_level, render=subdivide_level, overwrite=True)
            else:
                subdiv_mod_name = None

        wf_mod_name, _ = add_wireframe_modifier(cutter, thickness=thickness, overwrite=True)

        if subdiv_mod_name:
            print(f"Added Boolean modifier: {subdiv_mod_name}")
            apply_modifier(cutter, subdiv_mod_name)

        if wf_mod_name:
            print(f"Added Boolean modifier: {wf_mod_name}")
            apply_modifier(cutter, wf_mod_name)
        wireframe_paremeters['is_wireframed'] = True
        wireframe_paremeters['size'] = size
        wireframe_paremeters['pos_0'] = pos_0_list
        wireframe_paremeters['wire_type'] = cut_type
        wireframe_paremeters['rotation_euler'] = rotation_euler
        wireframe_paremeters['thickness'] = thickness
        wireframe_paremeters['subdivide_type'] = subdivide_type
        wireframe_paremeters['subdivide_level'] = subdivide_level
        wireframe_paremeters['material_name'] = mat_name
    else:
        print('Skip wireframe augmentation.')
        wireframe_paremeters['is_wireframed'] = False
    return wireframe_paremeters
