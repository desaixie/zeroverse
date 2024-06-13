# Zeroverse
Official code for arxiv paper [LRM-Zero: Training Large Reconstruction Models with Synthesized Data](https://arxiv.org), by Desai Xie, Sai Bi, Zhixin Shu, Kai Zhang, Zexiang Xu, Yi Zhou, Soren Pirk, Arie Kaufman, Xin Sun, Hao Tan, a collaboration between Adobe Research, Stony Brook University, and Kiel University.

This repository only contains the minimal code for procedurally synthesizing a single *Zeroverse* object.
To generate a large-scale *Zeroverse*dataset, you need to parallelly run the code with different seed for each object.
`create_shapes.py` is based on the SIGGRAPH 2018 (TOG) paper [Deep Image-Based Relighting from Optimal Sparse Samples](https://cseweb.ucsd.edu/~viscomp/projects/SIG18Relighting/PaperData/relight_paper.pdf) ([Project Webpage](https://cseweb.ucsd.edu/~viscomp/projects/SIG18Relighting/), [GitHub link](https://github.com/zexiangxu/Deep-Relighting)).

Here we use a publicly available material dataset, [MatSynth](https://gvecchio.com/matsynth/), which is different from the internal material dataset that we used for all experiment results reported in the paper.
This code is not extensively tested after switching to MatSynth, so please let us know if you encounter any issues.

## Environment Setup
- requires `python3.10`
```bash
sudo apt install libxi6 libsm6 libxext6
pip install datasets opencv-python pillow rich bpy==3.6.0
# install blender, then
export PATH=path/to/blender/:$PATH  # needs blender binary in addition to bpy to run zeroverse_rgba.py, augment_shapes.py and 2gltf2.py
```

## Randomly Sampled Parameters for Generated Shapes
- `create_shapes.py`
    - `uuid`: `str`
    - `sub_obj_num`: `int`, number of sub objects of this shape
    - `max_dim`: `float`
    - `sub_objs`: `[{}, {}, ...]`
        - `primitive_id`: `int`, `range(3)`
        - `axis_vals`: `array (3,)`
        - `translation`: `array (3,)`
        - `translation1`: `array (3,)`
        - `rotation`: `array (3,)`
        - `rotation1`: `array (3,)`
        - `height_fields`: `array (6, 36, 36)`
        - `material_id`: `int`, rgb2x material id
- `augment_shapes.py`
    - boolean augmentation
        - `is_augmented`: `boolean`
        - `size`: `random.uniform(0.3, 0.6)`, size of cutter
        - `pos_0`: `mathutils.Vector` of size 3, coordinate of a random vertex of the existing shape, used as the center of the cutter
        - `cut_type`: `random.choice(['sphere', 'cube', 'cone', 'cylinder', 'torus'])`
        - `rotation_euler`: `(random.uniform(0, 360) for _ in range(3))`
        - `thickness`: `random.uniform(0.01, 0.02)`, thickness of solidify after cutting
    - wireframe augmentation
        - `is_wireframed`: `boolean`
        - `size`: `random.uniform(0.3, 0.6)`, size of wireframe primitive
        - `thickness`: `random.uniform(0.01, 0.03)`, thickness of wireframe
        - `wire_type`: `random.choice(['sphere', 'cube', 'cone', 'cylinder', 'torus'])`
        - `subdivide_type`: `random.choice(['SIMPLE', 'CATMULL_CLARK'])`
        - `subdivide_level`: `random.choice([0, 1])`
        - `pos_0`: `mathutils.Vector` of size 3, coordinate of a random vertex of the existing shape, used as the center of the cutter
        - `rotation_euler`: `(random.uniform(0, 360) for _ in range(3))`
        - `material_name`: `str`, name of a material in the existing shape

## Single shape synthesizing and rendering
1. Synthesize objects in .obj and .mtl and convert to .glb: `python create_shapes.py --seed 0 --output_dir outputs/`
2. Optionally add boolean difference or wireframe augmentation and render .glb: `python zeroverse_rgba.py --seed 0 --object_path outputs/some_uuid/some_uuid.glb --output_dir outputs/some_uuid/views`
    - add `--boolean_probability 1.0` and/or `--wireframe_probability 1.0` to add these augmentations, which will be done right before rendering the views

## Zeroverse Dataset Generation
- You just need to parallelize the above single shape synthesizing process and run it for 800K times, each object with different seed. Then you will get your own Zeroverse dataset with 800K objects, match the size of Objaverse!

## Citation
If you find this code useful, please consider citing:
```
@article{xu2018deep,
  title={Deep image-based relighting from optimal sparse samples},
  author={Xu, Zexiang and Sunkavalli, Kalyan and Hadap, Sunil and Ramamoorthi, Ravi},
  journal={ACM Transactions on Graphics (TOG)},
  volume={37},
  number={4},
  pages={126},
  year={2018},
  publisher={ACM}
}
```