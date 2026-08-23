"""
Export synthesised view images mapped to their respective canvas/surface meshes.

This is one stage of the processing pipeline for https://github.com/mcmhsieh/Smile

SPDX-FileCopyrightText: 2026 Mark Hsieh
SPDX-License-Identifier: MIT
"""

import time
import pathlib
import shutil
import pickle

import numpy as np
import cv2
import PIL
import trimesh
import open3d as o3d

import IPython

from pipeline_server import start_pipeline_server, post_to_pipeline_server, get_queue_from_pipeline_server


# Note that the open3d.visualization.Visualizer class and its associated open3d.visualization.draw_geometries wrapper
# is the legacy way of visualizing open3d geometries.
# open3d.visualization.O3DVisualizer and its associated open3d.visualization.draw wrapper is powered by
# the https://github.com/google/filament rendering engine through the rendering Open3DScene / Scene classes.

# Note that for Open3D at least, the integrity of triangle mesh UV data is not maintained for some mesh operations
# such as cropping, decimation or vertex selection

def visualise_geometries(geometries, image_size, camera_intrinsic, lookat, up, front, zoom):
    """
    o3d.visualization.draw_geometries(geometries,
                                      lookat=lookat,
                                      up=up,
                                      front=front,
                                      zoom=zoom)
    """

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1024, height=768, left=200, top=200)

    camera_lines = o3d.geometry.LineSet.create_camera_visualization(view_width_px=image_size[0], view_height_px=image_size[1],
                                                                    intrinsic=camera_intrinsic,
                                                                    extrinsic=np.identity(4))
    camera_lines.paint_uniform_color([0, 0.5, 1])

    vis.add_geometry(camera_lines, reset_bounding_box=True)

    for geometry in geometries:
        if (isinstance(geometry, o3d.geometry.PointCloud)
            or isinstance(geometry, o3d.geometry.TriangleMesh) and not geometry.has_triangle_uvs()):
            geometry = geometry.crop(o3d.geometry.AxisAlignedBoundingBox([-100, -100, -100], [100, 100, 100]))
        vis.add_geometry(geometry, reset_bounding_box=True)

    ctr = vis.get_view_control()
    ctr.set_lookat(lookat)
    ctr.set_up(up)
    # vector from the lookat point to the camera
    ctr.set_front(front)
    ctr.set_zoom(zoom)
    ctr.set_constant_z_far(200.0)

    view_status = vis.get_view_status()
    view_status_time = time.time()
    visualisation_idle_timeout = 60 if IPython.get_ipython() is not None else 10
    while True:
        close_vis = not vis.poll_events()
        vis.update_renderer()
        new_view_status = vis.get_view_status()
        if new_view_status != view_status:
            view_status = new_view_status
            view_status_time = time.time()
        elif time.time() > view_status_time + visualisation_idle_timeout:
            close_vis = True
        if close_vis:
            break

    vis.destroy_window()


if __name__ == '__main__':

    working_subdir_config_path = pathlib.Path(r'../pipeline-workspace/working_subdir.txt')
    with open(working_subdir_config_path, 'r') as config_file:
        working_subdir = config_file.read().rstrip('\n')

    workspace_dirpath = pathlib.Path(r'../pipeline-workspace') / working_subdir
    input_source_dirpath = workspace_dirpath / 'view_synthesis'
    output_dirpath = workspace_dirpath / 'export_synthesised_views'

    if output_dirpath.exists():
        shutil.rmtree(output_dirpath)

    start_pipeline_server()
    post_to_pipeline_server((f'{working_subdir} / {output_dirpath.name}', 'waiting'))
    while True:
        pipeline_queue = get_queue_from_pipeline_server()
        print(pipeline_queue)
        if f'{working_subdir} / {input_source_dirpath.name}' not in pipeline_queue:
            break
        time.sleep(10)
    post_to_pipeline_server((f'{working_subdir} / {output_dirpath.name}', 'running'))
    print(get_queue_from_pipeline_server())

    # %%

    for input_path in sorted(input_source_dirpath.glob('*.*.pickle')):
        print(input_path)

        with open(input_path, 'rb') as pickle_file:
            data = pickle.load(pickle_file)
            synthetic_camera_extrinsic = data['synthetic_camera_extrinsic']
            camera_intrinsic_synthetic = data['camera_intrinsic_synthetic']
            filtered_up_model_synthetic_frame_img = data['filtered_up_model_synthetic_frame_img']
            vertices = data['vertices']
            triangles = data['triangles']
            vertex_colors = data['vertex_colors']

        canvas_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles))
        canvas_mesh.remove_vertices_by_mask(~np.all(np.isfinite(vertices), axis=1))
        canvas_mesh.transform(synthetic_camera_extrinsic)

        # Determine which mesh triangles require subdivision based on whether they map onto regions
        # of the synthetic image containing any void pixels.
        # Note that the synthetic image may include synthesised pixels outside the projected
        # canvas mesh boundaries because view synthesis upsamples the mapping score arrays.

        # TODO: Find area intersections of square tiles in a grid against triangles in a tessellation
        # (Querying points on a higher resolution grid can miss intersections
        # if triangles are small or have any acute angles)

        h, w = filtered_up_model_synthetic_frame_img.shape[:2]

        grid_scale = 8
        hj, wj = h * grid_scale, w * grid_scale

        camera_intrinsic_scaled = np.block([[camera_intrinsic_synthetic[:2, :2] * grid_scale, (camera_intrinsic_synthetic[:2, 2:] + 0.5) * grid_scale - 0.5], [0, 0, 1]])

        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(canvas_mesh))

        # Rays that exactly intersect vertices or edges may not register a raycasting hit due to floating point precision.
        # However in this case, since void pixels at the edge of any triangle have minimal impact on rendering,
        # there is no need to rectify this.
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(intrinsic_matrix=camera_intrinsic_scaled,
                                                                  extrinsic_matrix=np.identity(4),
                                                                  width_px=wj, height_px=hj)

        casted_rays = scene.cast_rays(rays)
        triangle_idxs = casted_rays['primitive_ids'].numpy()

        # Each pixel occupies the square between [u-0.5, u+0.5] & [v-0.5, v+0.5]
        # but the region in which it contributes to interpolation is [u-1, u+1] & [v-1, v+1]
        mask_image = cv2.resize((~np.all(np.isfinite(filtered_up_model_synthetic_frame_img), axis=-1)).astype(np.uint8),
                                (0, 0), fx=grid_scale, fy=grid_scale, interpolation=cv2.INTER_NEAREST)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (grid_scale + 1, grid_scale + 1))
        mask = cv2.dilate(mask_image, kernel, iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)

        subdiv_triangle_idxs = set(np.unique(triangle_idxs[mask])) - set([o3d.t.geometry.RaycastingScene.INVALID_ID])

        mapped_canvas_mesh = o3d.geometry.TriangleMesh(canvas_mesh)
        mapped_canvas_mesh.remove_triangles_by_index(list(subdiv_triangle_idxs))
        mapped_canvas_mesh.remove_unreferenced_vertices()

        subdiv_canvas_mesh = o3d.geometry.TriangleMesh(canvas_mesh)
        subdiv_canvas_mesh.remove_triangles_by_index(list(set(range(len(canvas_mesh.triangles))) - subdiv_triangle_idxs))
        subdiv_canvas_mesh.remove_unreferenced_vertices()

        subdiv_canvas_mesh = subdiv_canvas_mesh.subdivide_midpoint(number_of_iterations=3)


        # Determine which subdivided mesh triangles map onto regions
        # of the synthetic image containing any void pixels
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(subdiv_canvas_mesh))

        # Rays that exactly intersect vertices or edges may not register a raycasting hit due to floating point precision.
        # However in this case, since void pixels at the edge of any triangle have minimal impact on rendering,
        # there is no need to rectify this.
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(intrinsic_matrix=camera_intrinsic_scaled,
                                                                  extrinsic_matrix=np.identity(4),
                                                                  width_px=wj, height_px=hj)

        casted_rays = scene.cast_rays(rays)
        triangle_idxs = casted_rays['primitive_ids'].numpy()
        mask_triangle_idxs = set(np.unique(triangle_idxs[mask])) - set([o3d.t.geometry.RaycastingScene.INVALID_ID])

        mapped_subdiv_canvas_mesh = o3d.geometry.TriangleMesh(subdiv_canvas_mesh)
        mapped_subdiv_canvas_mesh.remove_triangles_by_index(list(mask_triangle_idxs))
        mapped_subdiv_canvas_mesh.remove_unreferenced_vertices()


        # Combine the original and subdivided triangles that map entirely onto regions
        # of the synthetic image with no void pixels
        trimmed_canvas_mesh = (mapped_canvas_mesh + mapped_subdiv_canvas_mesh).merge_close_vertices(eps=1e-3)

        projected_points = camera_intrinsic_synthetic @ np.array(trimmed_canvas_mesh.vertices).T
        projected_points = projected_points[:2, :] / projected_points[2, :]

        material_image = np.array(filtered_up_model_synthetic_frame_img)
        material_image[~np.isfinite(material_image)] = 127
        material_image = np.clip(material_image, 0, 255).astype(np.uint8)

        # baseColorTexture appears to be rendered with a fairly strong dependency on lighting and orientation
        # even if roughnessFactor = 1 and metallicFactor = 0, whereas emissiveTexture appears to be much less so.
        # When using emissiveTexture with metallicFactor = 0, there appears to be a small amount of lighting reflection,
        # which appears to diminish with metallicFactor = 1. This is probably because where metallicFactor = 0, the plastic shader
        # applies grey / white specular highlights, and where metallicFactor = 1, the metallic shader applies the base colour
        # as the diffuse colour of the metal, which has been set to black in the code below by pbr_material.baseColorFactor = [0, 0, 0, 0].
        pbr_material = trimesh.visual.material.PBRMaterial()
        pbr_material.baseColorFactor = [0, 0, 0, 0]
        #pbr_material.baseColorTexture = PIL.Image.fromarray(material_image)
        pbr_material.emissiveFactor = [1, 1, 1]
        pbr_material.emissiveTexture = PIL.Image.fromarray(material_image)
        pbr_material.roughnessFactor = 1.0
        pbr_material.metallicFactor = 1.0

        # uv origin is at bottom left of image for the GLB format
        uvs = np.vstack([(projected_points[0, :] + 0.5) / w, 1 - (projected_points[1, :] + 0.5) / h])

        tri_mesh = trimesh.Trimesh(vertices=np.array(trimmed_canvas_mesh.vertices),
                                   faces=np.array(trimmed_canvas_mesh.triangles),
                                   visual=trimesh.visual.TextureVisuals(uv=uvs.T, material=pbr_material))

        tri_mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))

        output_path = output_dirpath / (input_path.stem + '.trimesh.glb')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tri_mesh.export(str(output_path))

        if False:
            # uv origin is at bottom left of image for the O3DVisualizer / Filament rendering engine
            uvs = np.vstack([(projected_points[0, :] + 0.5) / w, 1 - (projected_points[1, :] + 0.5) / h])

            trimmed_canvas_mesh.triangle_uvs = o3d.utility.Vector2dVector(uvs[:, np.array(trimmed_canvas_mesh.triangles).flatten()].T)

            material = o3d.visualization.rendering.MaterialRecord()
            material.shader = 'defaultUnlit'
            material.albedo_img = o3d.geometry.Image(material_image)

            def vis_on_init(vis):
                vis.mouse_mode = o3d.visualization.gui.SceneWidget.Controls.ROTATE_MODEL

            """
            o3d.visualization.draw sometimes outputs:
                [Open3D INFO] Memory Statistics: (Device) (#Malloc) (#Free)
                [Open3D INFO] ---------------------------------------------
                [Open3D WARNING] CPU:0: 9 3 --> 6 with 116121600 total bytes
                [Open3D WARNING]     0x28d2914e040 @ 49766400 bytes
                [Open3D WARNING]     0x28d2e8b6040 @ 24883200 bytes
                [Open3D WARNING]     0x28d2c8de040 @ 8294400 bytes
                [Open3D WARNING]     0x28d2c0dd040 @ 8294400 bytes
                [Open3D WARNING]     0x28d2d8d1040 @ 16588800 bytes
                [Open3D WARNING]     0x28d2d0d7040 @ 8294400 bytes
                [Open3D INFO] ---------------------------------------------
            When using the mouse to rotate the model:
                [Open3D WARNING] max_bound {17.8268, 11.544, -20.4098} of bounding box is smaller than min_bound {-13.0449, -19.0061, -18.7131} in one or more axes. Fix input values to remove this warning.
            Gets stuck in an infinite loop when rerunning this module in IPython:
                [Open3D WARNING] GLFW Error: The GLFW library is not initialized
            """
            o3d.visualization.draw({'name': input_path.stem, 'geometry': trimmed_canvas_mesh, 'material': material},
                                   width=1600, height=1200,
                                   lookat=[0, 0, 1], eye=[0, 0, 0], up=[0, -1, 0],
                                   show_skybox=False, show_ui=False,
                                   on_init=vis_on_init)

        # uv origin is at top left of image for the legacy Visualizer
        uvs = np.vstack([(projected_points[0, :] + 0.5) / w, (projected_points[1, :] + 0.5) / h])

        trimmed_canvas_mesh.triangle_uvs = o3d.utility.Vector2dVector(uvs[:, np.array(trimmed_canvas_mesh.triangles).flatten()].T)

        trimmed_canvas_mesh.triangle_material_ids = o3d.utility.IntVector(np.zeros((len(trimmed_canvas_mesh.triangles),), dtype=int))
        trimmed_canvas_mesh.textures = [o3d.geometry.Image(material_image)]

        if False:
            output_path = output_dirpath / (input_path.stem + '.o3d.glb')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # [Open3D WARNING] This file format does not support writing textures and uv coordinates. Consider using .obj
            o3d.io.write_triangle_mesh(str(output_path), trimmed_canvas_mesh)
        if False:
            output_path = output_dirpath / (input_path.stem + '.o3d.obj')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # A .mtl material file is written separately to the .obj file
            o3d.io.write_triangle_mesh(str(output_path), trimmed_canvas_mesh)

        visualise_geometries([trimmed_canvas_mesh],
                             material_image.shape[1::-1],
                             camera_intrinsic_synthetic,
                             lookat=[0, 0, 8],
                             up=[0, -1, 0],
                             front=[0, 0, -8],
                             zoom=1.0)

    # %%

    post_to_pipeline_server((f'{working_subdir} / {output_dirpath.name}', None))
