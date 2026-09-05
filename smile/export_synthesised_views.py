"""
Export synthesised view images mapped to their respective canvas/surface meshes.
Produce interactive plots and HTML documents for viewing mapped frame images and frame projection boundaries.

This is one stage of the processing pipeline for https://github.com/mcmhsieh/Smile

SPDX-FileCopyrightText: 2026 Mark Hsieh
SPDX-License-Identifier: MIT
"""

import time
import pathlib
import shutil
import pickle
import base64
import functools
import gzip
import io
import json
import collections

import numpy as np
import cv2
import PIL
import trimesh
import open3d as o3d
import shapely

import IPython
spyder_ide = IPython.get_ipython().__class__.__name__ == 'SpyderShell'

import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
if not spyder_ide:
    # TODO: check whether the system display scaling setting needs to be taken into account
    plt.rcParams['figure.dpi'] = 80.0
    if hasattr(IPython.get_ipython(), 'run_line_magic'):
        IPython.get_ipython().run_line_magic('matplotlib', 'qt')

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
    image_source_dirpath = workspace_dirpath / 'calc_sequential_flow_and_blur'
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

    input_path = workspace_dirpath / 'stitch_key_frames' / 'stitched_key_frames.pickle'
    with open(input_path, 'rb') as pickle_file:
        data = pickle.load(pickle_file)
        key_frame_indices = data['key_frame_indices']
        key_frame_motion_blurs = data['key_frame_motion_blurs']
        triangulated_idxs_weights = data['triangulated_idxs_weights']
        key_frame_image_sample_points = data['key_frame_image_sample_points']
        key_frame_image_triangulated_point_idxs = data['key_frame_image_triangulated_point_idxs']
        cross_stitch_disparity_confidence_maps = data['cross_stitch_disparity_confidence_maps']
        camera_extrinsics = data['camera_extrinsics']
        camera_intrinsic = data['camera_intrinsic']
        model_triangulated_points = data['model_triangulated_points']
        post_optimise_triangulated_idxs_mask = data['post_optimise_triangulated_idxs_mask']

    # %%

    frame_images = []
    for frame_index, frame_time in key_frame_indices:
        filename = f'{frame_time.strftime("%Y%m%d-%H%M%S%f")}.{frame_index:03d}.resized.png'
        frame_images.append(cv2.cvtColor(cv2.imread(image_source_dirpath / filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH), cv2.COLOR_BGR2RGB))

    image_sizes = set([img.shape[1::-1] for img in frame_images])
    assert len(image_sizes) == 1
    image_size = image_sizes.pop()

    # %%

    def remove_indentation(html):
        # https://developer.mozilla.org/en-US/docs/Web/CSS/Guides/Text/Whitespace
        # all spaces and tabs immediately before and after a line break are ignored
        line_indentations = np.array([len(line.rstrip(' ')) - len(line.strip(' ')) for line in html.splitlines()])
        indentation = np.min(line_indentations[line_indentations > 0])
        # IPython runcell() appears to add leading spaces as indentation to all lines, including
        # multi-line strings and empty lines.
        # Workaround by removing trailing spaces.
        return '\n'.join([line[indentation:].rstrip(' ') for line in html.splitlines()]).strip()

    # %%

    for input_path in sorted(input_source_dirpath.glob('*.pickle')):
        print(input_path)

        with open(input_path, 'rb') as pickle_file:
            data = pickle.load(pickle_file)
            synthetic_camera_extrinsic = data['synthetic_camera_extrinsic']
            camera_intrinsic_synthetic = data['camera_intrinsic_synthetic']
            filtered_up_model_synthetic_frame_img = data['filtered_up_model_synthetic_frame_img']
            vertices = data['vertices']
            triangles = data['triangles']
            vertex_colors = data['vertex_colors']
            up_model_frames_idxs = data['up_model_frames_idxs']
            up_model_cmap = data['up_model_cmap']
            up_max_model_mapping_scores = data['up_max_model_mapping_scores']

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


        if False:
            # TODO: Determine why three.js renders the emissive texture so darkly
            import trimesh.viewer

            trimesh_camera = trimesh.scene.cameras.Camera(name='camera', resolution=material_image.shape[:2],
                                                          focal=np.diag(camera_intrinsic_synthetic)[:2])
            trimesh_scene = trimesh.Scene(geometry=[tri_mesh],  camera=trimesh_camera)

            output_path = output_dirpath / (input_path.stem + '.trimesh.scene.html')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as html_file:
                html_file.write(trimesh.viewer.notebook.scene_to_html(trimesh_scene))


        output_path = output_dirpath / (input_path.stem + '.trimesh.glb')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tri_mesh_glb_data = tri_mesh.export(str(output_path))

        # https://doc.babylonjs.com/features/featuresDeepDive/babylonViewer/
        html = remove_indentation(r"""
          <!DOCTYPE html>
          <html>
            <head>
              <title>Babylon Viewer</title>
              <meta charset="UTF-8"/>
              <script type="module" src="https://cdn.jsdelivr.net/npm/@babylonjs/viewer@9.22.2/dist/babylon-viewer.esm.min.js"></script>
              <style>
                html, body { width: 100%; height: 100%; padding: 0; margin: 0; overflow: hidden; }
                body { background: repeating-conic-gradient(#d2d2d2 0% 25%, white 0% 50%) 50% / 16px 16px }
              </style>
            </head>
            <body>
              <babylon-viewer source="data:;base64,&&B64_GLB_DATA&&"
                camera-orbit="1.571 1.571 20" camera-target="0 0 -10">
              </babylon-viewer>
              <script>
                const viewerElement = document.querySelector("babylon-viewer");
                viewerElement.addEventListener("viewerready", () => {
                  const scene = viewerElement.viewerDetails.scene;
                  const camera = viewerElement.viewerDetails.camera;
                  camera.fov = 80 / 180 * 3.142;
                  let frameTime = 0;
                  function cameraOrbit(frameTime) {
                    const r = (0.5 + 9.5 / (1 + Math.exp(-(frameTime - 40) / 30 * 6))) / 180 * 3.142;
                    const angle = frameTime * 2 * 3.142 / 20;
                    return [r * Math.sin(angle), r * Math.cos(angle)];
                  }
                  viewerElement.addEventListener("click", (event) => {
                    frameTime = 0;
                  });
                  scene.onBeforeRenderObservable.add(() => {
                    const deltaTime = 1e-3 * scene.getEngine().getDeltaTime();
                    camera.alpha += cameraOrbit(frameTime + deltaTime)[0] - cameraOrbit(frameTime)[0];
                    camera.beta += cameraOrbit(frameTime + deltaTime)[1] - cameraOrbit(frameTime)[1];
                    frameTime += deltaTime;
                  });
                });
              </script>
            </body>
          </html>
        """).replace('&&B64_GLB_DATA&&', base64.b64encode(tri_mesh_glb_data).decode('utf-8'))

        output_path = output_dirpath / (input_path.stem + '.trimesh.html')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as html_file:
            html_file.write(html)


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

    projected_polygon_coords = collections.defaultdict(dict)
    for input_path in sorted(input_source_dirpath.glob('*.pickle')):
        with open(input_path, 'rb') as pickle_file:
            data = pickle.load(pickle_file)
            synthetic_camera_extrinsic = data['synthetic_camera_extrinsic']
            camera_intrinsic_synthetic = data['camera_intrinsic_synthetic']
            filtered_up_model_synthetic_frame_img = data['filtered_up_model_synthetic_frame_img']
            vertices = data['vertices']
            triangles = data['triangles']
            vertex_colors = data['vertex_colors']
            up_model_frames_idxs = data['up_model_frames_idxs']
            up_model_cmap = data['up_model_cmap']
            up_max_model_mapping_scores = data['up_max_model_mapping_scores']

        # np.unique() returns sorted unique elements
        secondary_frame_idxs = np.unique(up_model_frames_idxs)

        canvas_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles))
        canvas_mesh.remove_vertices_by_mask(~np.all(np.isfinite(vertices), axis=1))
        canvas_mesh.compute_vertex_normals()
        valid_vertices = np.array(canvas_mesh.vertices)
        valid_vertex_normals = np.array(canvas_mesh.vertex_normals)

        ref_projected_points = camera_intrinsic_synthetic @ (synthetic_camera_extrinsic[:3, :3] @ valid_vertices.T + synthetic_camera_extrinsic[:3, 3:])
        ref_projected_points = ref_projected_points[:2, :] / ref_projected_points[2, :]

        for secondary_frame_idx in secondary_frame_idxs:
            secondary_camera_extrinsic = camera_extrinsics[secondary_frame_idx]

            vertex_points = secondary_camera_extrinsic[:3, :3] @ valid_vertices.T + secondary_camera_extrinsic[:3, 3:]
            vertex_normals = secondary_camera_extrinsic[:3, :3] @ valid_vertex_normals.T

            secondary_projected_points = camera_intrinsic @ vertex_points
            secondary_projected_points = secondary_projected_points[:2, :] / secondary_projected_points[2, :]

            camera_rays = vertex_points / np.clip(np.linalg.norm(vertex_points, axis=0), 1e-6, np.inf)
            normal_ray_alignment = np.sum(camera_rays * vertex_normals, axis=0)

            w, h = image_size
            inlier_mask = ((secondary_projected_points[0, :] > -0.5) & (secondary_projected_points[0, :] < w - 0.5)
                           & (secondary_projected_points[1, :] > -0.5) & (secondary_projected_points[1, :] < h - 0.5)
                           & (normal_ray_alignment < 0))

            projected_polygon = shapely.concave_hull(shapely.MultiPoint(ref_projected_points[:, inlier_mask].T), ratio=0.2)
            projected_polygon_coords[input_path.stem][secondary_frame_idx] = np.array(projected_polygon.boundary.coords)

    # %%

    for input_path in sorted(input_source_dirpath.glob('*.pickle')):
        print(input_path)

        with open(input_path, 'rb') as pickle_file:
            data = pickle.load(pickle_file)
            synthetic_camera_extrinsic = data['synthetic_camera_extrinsic']
            camera_intrinsic_synthetic = data['camera_intrinsic_synthetic']
            filtered_up_model_synthetic_frame_img = data['filtered_up_model_synthetic_frame_img']
            vertices = data['vertices']
            triangles = data['triangles']
            vertex_colors = data['vertex_colors']
            up_model_frames_idxs = data['up_model_frames_idxs']
            up_model_cmap = data['up_model_cmap']
            up_max_model_mapping_scores = data['up_max_model_mapping_scores']

        # np.unique() returns sorted unique elements
        secondary_frame_idxs = np.unique(up_model_frames_idxs)

        fig_elements = {}

        fig = plt.figure(f'Interactive frame mapping {input_path.stem}', figsize=(24, 12))
        fig.clf()
        fig_ax1 = plt.subplot(1, 2, 1)
        fig_ax2 = plt.subplot(1, 2, 2)

        fig_ax1.cla()
        fig_ax1.set_facecolor('grey')
        fig_ax1.imshow(np.clip(np.require(filtered_up_model_synthetic_frame_img, dtype=np.float32) / 255, 0, 1))
        #fig_ax1.imshow(up_model_frames_idxs, cmap=up_model_cmap, vmin=0, vmax=up_model_cmap.N, interpolation_stage='rgba', alpha=0.2)
        fig_ax2.cla()
        fig_elements['sec_img'] = fig_ax2.imshow(np.require(frame_images[secondary_frame_idxs[0]], dtype=np.uint8))
        fig_elements['sec_title'] = fig_ax2.set_title(f'frame idx: {secondary_frame_idxs[0]}')

        projected_polygons = {frame_idx: fig_ax1.plot(*coords.T, color='b')
                              for frame_idx, coords in projected_polygon_coords[input_path.stem].items()}

        def fig_on_resize(fig, fig_elements, projected_polygons, event):
            for polygon_boundary_lines in projected_polygons.values():
                for line in polygon_boundary_lines:
                    line.set_alpha(0)
            fig_elements['sec_title'].set_alpha(0)
            fig.tight_layout()
            fig.canvas.draw()
            fig_elements['background'] = fig.canvas.copy_from_bbox(fig.bbox)
            fig_elements['sec_title'].set_alpha(1)
            fig.canvas.draw_idle()

        def fig_on_motion(fig, fig_ax1, fig_ax2, fig_elements,
                          filtered_up_model_synthetic_frame_img, up_model_frames_idxs, projected_polygons, event):
            fig_scaling = np.sqrt(np.linalg.det(fig_ax1.transAxes.get_matrix()[:2, :2])) * 1e-3
            for polygon_boundary_lines in projected_polygons.values():
                for line in polygon_boundary_lines:
                    line.set_alpha(0)
                    line.set_linewidth(3 * fig_scaling)

            secondary_frame_idx = None
            if event.inaxes == fig_ax1:
                mouse_xy = np.array([event.xdata, event.ydata])
                x, y = np.round(mouse_xy).astype(int)
                if np.all(np.isfinite(filtered_up_model_synthetic_frame_img[y, x, :]), axis=-1):
                    secondary_frame_idx = up_model_frames_idxs[y, x]
                    for line in projected_polygons[secondary_frame_idx]:
                        line.set_alpha(0.5)
                    fig_elements['sec_img'].set_data(frame_images[secondary_frame_idx])
                    fig_elements['sec_title'].set_text(f'frame idx: {secondary_frame_idx}')

            #fig.canvas.draw()
            #fig.canvas.draw_idle()

            fig.canvas.restore_region(fig_elements['background'])
            for polygon_boundary_lines in projected_polygons.values():
                for line in polygon_boundary_lines:
                    fig_ax1.draw_artist(line)
            fig_ax2.draw_artist(fig_elements['sec_img'])
            fig_ax2.draw_artist(fig_elements['sec_title'])
            fig.canvas.blit(fig.bbox)
            fig.canvas.flush_events()

        for cid in [cid for signal, cid_ref_map in fig.canvas.callbacks.callbacks.items() for cid in cid_ref_map]:
            fig.canvas.mpl_disconnect(cid)

        fig_on_resize_callback = functools.partial(fig_on_resize, fig, fig_elements, projected_polygons)
        fig_on_motion_callback = functools.partial(fig_on_motion, fig, fig_ax1, fig_ax2, fig_elements,
                                                   filtered_up_model_synthetic_frame_img, up_model_frames_idxs, projected_polygons)
        cid_resize = fig.canvas.mpl_connect('resize_event', fig_on_resize_callback)
        cid_motion = fig.canvas.mpl_connect('motion_notify_event', fig_on_motion_callback)
        cid_axes_leave = fig.canvas.mpl_connect('axes_leave_event', fig_on_motion_callback)
        cid_fig_leave = fig.canvas.mpl_connect('figure_leave_event', fig_on_motion_callback)

        fig_on_resize_callback(None)

    # %%

    # =====================================================================
    # WebP image embedding
    # =====================================================================

    def image_to_webp_data_uri(image: PIL.Image, quality: int = 90) -> str:
        """
        Convert an image to WebP and return a base64 data URI.
        """

        assert image.mode == 'RGB'
        buffer = io.BytesIO()
        image.save(buffer, format='WEBP', quality=quality, method=6)
        encoded = base64.b64encode(buffer.getvalue()).decode('ascii')

        return 'data:image/webp;base64,' + encoded

    # =====================================================================
    # HTML generator
    # =====================================================================

    def generate_panorama_html(panorama_image: PIL.Image,
                               map_image: PIL.Image,
                               frame_images: list[PIL.Image],
                               key_frame_hulls: list[list[tuple[float, float]]],
                               webp_quality: int = 90):
        """
        Generate a self-contained panorama viewer.
        """

        # ---------------------------------------------------------------
        # Panorama & map and key frame dimensions.
        # ---------------------------------------------------------------

        if panorama_image.size != map_image.size:
            raise ValueError(f'Panorama size {panorama_image.size} does not match map size {map_image.size}')

        width, height = panorama_image.size

        frame_width, frame_height = np.max([image.size for image in frame_images], axis=0)

        # ---------------------------------------------------------------
        # Key frame hulls.
        # ---------------------------------------------------------------

        hulls_json = json.dumps(key_frame_hulls, separators=(',', ':'))

        # ---------------------------------------------------------------
        # Panorama -> WebP.
        # ---------------------------------------------------------------

        panorama_uri = image_to_webp_data_uri(panorama_image, webp_quality)

        # ---------------------------------------------------------------
        # Key frames -> WebP.
        # ---------------------------------------------------------------

        key_frame_uris = [image_to_webp_data_uri(image, webp_quality) for image in frame_images]
        key_frame_json = json.dumps(key_frame_uris, separators=(',', ':'))

        # ---------------------------------------------------------------
        # gzip map.
        # ---------------------------------------------------------------

        map_compressed = gzip.compress(map_image.tobytes(), compresslevel=9)

        # ---------------------------------------------------------------
        # Base64 for embedding in JavaScript.
        # ---------------------------------------------------------------

        map_base64 = base64.b64encode(map_compressed).decode('ascii')

        # ---------------------------------------------------------------
        # HTML.
        # ---------------------------------------------------------------

        html_document = remove_indentation(f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Synthesised Panorama Key Frame Viewer</title>
            <style>

            body {{
              margin: 0;
              padding: 0px 20px 0px 20px;
              overflow: hidden;
              background: #222;
              color: white;
              font-family: Arial, sans-serif;
            }}

            #container {{
              display: flex;
              align-items: flex-start;
              gap: 20px;
              width: 100%;
              height: calc(100vh - 7em);
            }}

            /*
             * updateItemSizes() calculates and sets the final displayed width and height of this item.
             */
            #panoramaItem {{
              flex: 0 1 auto;
            }}

            /*
             * Container for the panorama and the mapped key-frame convex hull polygon SVG overlay.
             * They maintain the same dimensions by sharing the same grid cell.
             */
            #panoramaContainer {{
              display: grid;
              position: relative;
              width: 100%;
              height: auto;
            }}

            #panorama {{
              display: block;
              grid-column: 1;
              grid-row: 1;
              width: 100%;
              height: 100%;
              object-fit: contain;
              margin: 0;
              cursor: crosshair;
              touch-action: none;
              user-select: none;
              -webkit-user-drag: none;
            }}

            #hullOverlay {{
              grid-column: 1;
              grid-row: 1;
              width: 100%;
              height: 100%;
              object-fit: contain;
              margin: 0;
              pointer-events: none;
            }}

            #hullPolygon {{
              fill: rgba(0, 100, 255, 0.05);
              stroke: rgba(0, 120, 255, 0.75);
              stroke-width: 2;
              vector-effect: non-scaling-stroke;
              display: none;
            }}

            /*
             * updateItemSizes() calculates and sets the final displayed width and height of this item.
             */
            #keyFrameItem {{
              flex: 0 1 auto;
              margin: 0;
              padding: 5px;
              box-sizing: border-box;
              background: #111;
              border: 1px solid #555;
            }}

            #keyFrame {{
              display: none;
              width: 100%;
              height: 100%;
              object-fit: contain;
              margin: 0;
            }}

            #placeholder {{
              color: #777;
            }}

            #info {{
              display: block;
              color: #aaa;
              font-family: monospace;
              margin: 0;
            }}

            #spacerItem {{
              flex: 1 1 auto;
            }}

            /*
             * On portrait displays, stack the panorama and key frame items vertically.
             */
            @media (orientation: portrait) {{

              body {{
                padding: 0px 10px 0px 10px;
              }}

              #container {{
                flex-direction: column;
                gap: 10px;
                height: calc(100vh - 6em);
              }}

              #panoramaItem {{
                width: 100%;
                height: auto;
              }}

              #keyFrameItem {{
                width: 100%;
                height: auto;
              }}

              #info {{
                display: none;
              }}

            }}

            </style>
            </head>

            <body>

            <h2>Synthesised Panorama Key Frame Viewer</h2>

            <div id="container">

              <!-- =========================================================
                   Panorama item
                   ========================================================= -->

              <div id="panoramaItem">

                <div id="panoramaContainer">

                  <img id="panorama" src="{panorama_uri}" width="{width}" height="{height}" alt="Panorama">

                  <!--
                       SVG coordinate system exactly matches the native panorama coordinate system.
                  -->

                  <svg id="hullOverlay" viewBox="0 0 {width} {height}" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
                    <polygon id="hullPolygon" points=""/>
                  </svg>

                </div>

                <div id="info">
                  Loading map...
                </div>

              </div>

              <!-- =========================================================
                   Key frame item
                   ========================================================= -->

              <div id="keyFrameItem">
                <img id="keyFrame" alt="Key frame">
                <span id="placeholder">Move, tap or drag over the panorama image to view key frames.</span>
              </div>

              <!-- =========================================================
                   Spacer item
                   ========================================================= -->

              <div id="spacerItem">
              </div>

            </div>

            <script>

            "use strict";

            /* =================================================================
               Panorama & map dimensions
               ================================================================= */

            const sourceWidth = {width};
            const sourceHeight = {height};

            /* =================================================================
               Key-frame WebP images
               ================================================================= */

            const keyFrameUris = {key_frame_json};
            const keyFrameWidth = {frame_width};
            const keyFrameHeight = {frame_height};

            /* =================================================================
               Key-frame convex hulls
               =================================================================

               keyFrameHulls[index] contains: [[x1, y1], [x2, y2], ...]
               Coordinates are in native panorama pixel coordinates.
            */

            const keyFrameHulls = {hulls_json};

            /* =================================================================
               Compressed encoded map
               ================================================================= */

            const mapBase64 = "{map_base64}";

            /* =================================================================
               Base64 decoder
               ================================================================= */

            function base64ToUint8Array(base64) {{
              const binary = atob(base64);
              const bytes = new Uint8Array(binary.length);

              for (let i = 0; i < binary.length; ++i) {{
                bytes[i] = binary.charCodeAt(i);
              }}

              return bytes;
            }}

            /* =================================================================
               Decode and decompress map
               ================================================================= */

            async function decodeMap() {{
              /*
               * Base64 -> gzip data.
               */
              const compressed = base64ToUint8Array(mapBase64);

              /*
               * Decompress gzip data.
               */
              const stream = new Blob([compressed]).stream().pipeThrough(new DecompressionStream("gzip"));
              const buffer = await new Response(stream).arrayBuffer();

              return new Int32Array(buffer);
            }}

            /* =================================================================
               DOM elements
               ================================================================= */

            const container = document.getElementById("container");
            const panoramaItem = document.getElementById("panoramaItem");
            const panoramaContainer = document.getElementById("panoramaContainer");
            const panorama = document.getElementById("panorama");
            const hullOverlay = document.getElementById("hullOverlay");
            const hullPolygon = document.getElementById("hullPolygon");
            const keyFrameItem = document.getElementById("keyFrameItem");
            const keyFrame = document.getElementById("keyFrame");
            const placeholder = document.getElementById("placeholder");
            const info = document.getElementById("info");
            const spacerItem = document.getElementById("spacerItem");

            /* =================================================================
               Viewer state
               ================================================================= */

            let mapValues = null;
            let previousIndex = null;

            /* =================================================================
               Display convex hull
               ================================================================= */

            function displayHull(keyFrameIndex) {{
              /*
               * Explicitly handle -1.
               *
               * There is no key frame and therefore no polygon.
               */
              if (keyFrameIndex === -1) {{
                hideHull();
                return;
              }}

              const hull = keyFrameHulls[keyFrameIndex];

              /*
               * No hull available.
               */
              if (!hull || hull.length < 3) {{
                hideHull();
                return;
              }}

              /*
               * Convert:
               *     [[x1,y1], [x2,y2], [x3,y3]]
               * to:
               *     "x1,y1 x2,y2 x3,y3"
               */
              const points = hull.map(function(point) {{
                return (point[0] + "," + point[1]);
              }}).join(" ");

              hullPolygon.setAttribute("points", points);

              hullPolygon.style.display = "block";
            }}

            /* =================================================================
               Hide convex hull
               ================================================================= */

            function hideHull() {{
              hullPolygon.style.display = "none";
              hullPolygon.setAttribute("points", "");
            }}

            /* =================================================================
               Display key frame
               ================================================================= */

            function displayKeyFrame(keyFrameIndex) {{
              /*
               * -1 explicitly means no key frame.
               */
              if (keyFrameIndex === -1) {{
                hideKeyFrame();
                return;
              }}

              /*
               * Validate index.
               */
              if (!Number.isInteger(keyFrameIndex) || keyFrameIndex < 0 || keyFrameIndex >= keyFrameUris.length) {{
                hideKeyFrame();
                return;
              }}

              const uri = keyFrameUris[keyFrameIndex];

              if (!uri) {{
                hideKeyFrame();
                return;
              }}

              keyFrame.src = uri;
              keyFrame.style.display = "block";
              placeholder.style.display = "none";
            }}

            /* =================================================================
               Hide key frame
               ================================================================= */

            function hideKeyFrame() {{
              keyFrame.style.display = "none";
              keyFrame.removeAttribute("src");
              placeholder.style.display = "block";
            }}

            /* =================================================================
               Load map
               ================================================================= */

            decodeMap().then(function(map) {{
              mapValues = map;
              info.textContent = "Map loaded. Move, tap or drag over the panorama image to view key frames.";
            }}).catch(function(error) {{
              console.error(error);
              info.textContent = "Error loading map.";
            }});

            /* =================================================================
               Calculate and set displayed panorama and key frame item dimensions.
               ================================================================= */

            function updateItemSizes() {{
              let frameDisplayWidth;
              let frameDisplayHeight;
              let sourceDisplayWidth;
              let sourceDisplayHeight;

              const frameAspectRatio = keyFrameWidth / keyFrameHeight;
              const sourceAspectRatio = sourceWidth / sourceHeight;

              /*
               * https://developer.mozilla.org/en-US/docs/Web/CSS/Reference/At-rules/@media/orientation
               * portrait: viewport height >= width
               * landscape: viewport width > height
               * Note: This feature does not correspond to device orientation.
               */
              if (window.matchMedia("(orientation: portrait)").matches) {{
                /*
                 * Allocate 2/3 of the available height to the key frame item and the remainder to the panorama.
                 * Constrain their dimensions to maintain their respective image aspect ratios.
                 */
                const availableWidth = Math.max(container.clientWidth, 100);
                const availableHeight = Math.max(panoramaItem.clientHeight + keyFrameItem.clientHeight
                                                 + spacerItem.clientHeight + 10, 100);

                const frameAvailableHeight = Math.max(2 / 3 * availableHeight, 100);

                frameDisplayWidth = Math.min(availableWidth, keyFrameWidth);
                frameDisplayHeight = frameDisplayWidth / frameAspectRatio;

                if (frameDisplayHeight > frameAvailableHeight) {{
                  frameDisplayHeight = frameAvailableHeight;
                  frameDisplayWidth = frameDisplayHeight * frameAspectRatio;
                }}

                const sourceAvailableHeight = Math.max(availableHeight - frameDisplayHeight, 100);

                sourceDisplayWidth = Math.min(availableWidth, sourceWidth);
                sourceDisplayHeight = sourceDisplayWidth / sourceAspectRatio;

                if (sourceDisplayHeight > sourceAvailableHeight) {{
                  sourceDisplayHeight = sourceAvailableHeight;
                  sourceDisplayWidth = sourceDisplayHeight * sourceAspectRatio;
                }}
              }} else {{
                /*
                 * Allocate 2/3 of the available width to the key frame item and the remainder to the panorama.
                 * Constrain their dimensions to maintain their respective image aspect ratios.
                 */
                const availableHeight = Math.max(container.clientHeight, 100);
                const availableWidth = Math.max(panoramaItem.clientWidth + keyFrameItem.clientWidth
                                                + spacerItem.clientWidth + 20, 100);

                const frameAvailableWidth = Math.max(2 / 3 * availableWidth, 100);

                frameDisplayHeight = Math.min(availableHeight, keyFrameHeight);
                frameDisplayWidth = frameDisplayHeight * frameAspectRatio;

                if (frameDisplayWidth > frameAvailableWidth) {{
                  frameDisplayWidth = frameAvailableWidth;
                  frameDisplayHeight = frameDisplayWidth / frameAspectRatio;
                }}

                const sourceAvailableWidth = Math.max(availableWidth - frameDisplayWidth, 100);

                sourceDisplayHeight = Math.min(availableHeight, sourceHeight);
                sourceDisplayWidth = sourceDisplayHeight * sourceAspectRatio;

                if (sourceDisplayWidth > sourceAvailableWidth) {{
                  sourceDisplayWidth = sourceAvailableWidth;
                  sourceDisplayHeight = sourceDisplayWidth / sourceAspectRatio;
                }}
              }}

              /*
               * Set the item dimensions.
               */
              keyFrameItem.style.width = Math.round(frameDisplayWidth) + "px";
              keyFrameItem.style.height = Math.round(frameDisplayHeight) + "px";
              panoramaItem.style.width = Math.round(sourceDisplayWidth) + "px";
              panoramaItem.style.height = Math.round(sourceDisplayHeight) + "px";
            }}

            /*
             * Update item sizes after all page resources have loaded, on browser resizing
             * and container resizing.
             */
            window.addEventListener("load", updateItemSizes);
            window.addEventListener("resize", updateItemSizes);
            const panoramaResizeObserver = new ResizeObserver(function() {{
              updateItemSizes();
            }});
            panoramaResizeObserver.observe(container);

            /* =================================================================
               Pointer position event handler
               ================================================================= */

            function updateFromPointer(event) {{
              if (!mapValues) {{
                return;
              }}

              /*
               * Bounding rectangle of the displayed panorama.
               */
              const rect = panorama.getBoundingClientRect();

              /*
               * Pointer position relative to displayed image.
               */
              const displayX = event.clientX - rect.left;
              const displayY = event.clientY - rect.top;

              /*
               * Ignore pointer positions outside the displayed panorama.
               */
              if (displayX < 0 || displayX >= rect.width ||
                  displayY < 0 || displayY >= rect.height) {{
                return;
              }}

              /*
               * Convert displayed coordinates to native panorama coordinates.
               */
              let x = Math.round(displayX * sourceWidth / rect.width);
              let y = Math.round(displayY * sourceHeight / rect.height);

              /*
               * Clamp coordinates.
               */
              x = Math.max(0, Math.min(sourceWidth - 1, x));
              y = Math.max(0, Math.min(sourceHeight - 1, y));

              /*
               * O(1) map lookup.
               */
              const mapPosition = y * sourceWidth + x;
              const keyFrameIndex = mapValues[mapPosition];

              info.textContent = "x=" + x + "  y=" + y;
              if (keyFrameIndex >= 0) {{
                info.textContent += "  key_frame=" + keyFrameIndex;
              }}

              /*
               * Don't update the display if the map value hasn't changed.
               */
              if (keyFrameIndex === previousIndex) {{
                return;
              }}

              previousIndex = keyFrameIndex;

              displayHull(keyFrameIndex);
              displayKeyFrame(keyFrameIndex);
            }}

            /* =================================================================
               Mouse, touch and stylus movement
               ================================================================= */

            panorama.addEventListener("pointermove", function(event) {{
              updateFromPointer(event);
            }});

            /* =================================================================
               Tap or pointer press
               ================================================================= */

            panorama.addEventListener("pointerdown", function(event) {{
              updateFromPointer(event);
            }});

            /* =================================================================
               Mouse leaves panorama
               ================================================================= */

            panorama.addEventListener("pointerleave", function(event) {{
              if (event.pointerType == "mouse") {{
                hideHull();
                hideKeyFrame();
                previousIndex = null;
                info.textContent = "Move, tap or drag over the panorama image to view key frames.";
              }}
            }});

            </script>
            </body>
            </html>
        """)

        # ---------------------------------------------------------------
        # Report sizes.
        # ---------------------------------------------------------------

        print('Panorama:')
        print(f'  dimensions: {width} x {height}')
        print('Map:')
        print(f'  gzip size:   {len(map_compressed):,} bytes')
        print(f'  Base64 size: {len(map_base64):,} characters')

        return html_document

    for input_path in sorted(input_source_dirpath.glob('*.pickle')):
        print(input_path)

        with open(input_path, 'rb') as pickle_file:
            data = pickle.load(pickle_file)
            synthetic_camera_extrinsic = data['synthetic_camera_extrinsic']
            camera_intrinsic_synthetic = data['camera_intrinsic_synthetic']
            filtered_up_model_synthetic_frame_img = data['filtered_up_model_synthetic_frame_img']
            vertices = data['vertices']
            triangles = data['triangles']
            vertex_colors = data['vertex_colors']
            up_model_frames_idxs = data['up_model_frames_idxs']
            up_model_cmap = data['up_model_cmap']
            up_max_model_mapping_scores = data['up_max_model_mapping_scores']

        # Hull vertices must be ordered around the perimeter
        key_frame_hulls = [[tuple(row) for row in projected_polygon_coords[input_path.stem].get(frame_idx, [])]
                           for frame_idx in range(len(frame_images))]

        if True:
            # Note that the cursor becomes invisible when the background colour is (127, 127, 127)
            panorama_image = np.array(filtered_up_model_synthetic_frame_img)
            panorama_image[~np.all(np.isfinite(panorama_image), axis=-1)] = np.array([[[51, 51, 76]]])
            panorama_image = np.clip(panorama_image, 0, 255).astype(np.uint8)
        else:
            # Switching from the panorama image to the key frame map can be useful for testing & debugging
            panorama_image = (up_model_cmap(up_model_frames_idxs)[:, :, :3] * 255).astype(np.uint8)
            panorama_image[~np.all(np.isfinite(filtered_up_model_synthetic_frame_img), axis=-1)] = np.array([[[255, 255, 255]]])
        panorama_image = PIL.Image.fromarray(panorama_image)

        # Negative map image values denote no key frame mapping
        map_image = np.array(up_model_frames_idxs)
        map_image[~np.all(np.isfinite(filtered_up_model_synthetic_frame_img), axis=-1)] = -1
        map_image = PIL.Image.fromarray(map_image)

        panorama_html = generate_panorama_html(panorama_image=panorama_image, map_image=map_image,
                                               frame_images=[PIL.Image.fromarray(image) for image in frame_images],
                                               key_frame_hulls=key_frame_hulls,
                                               webp_quality=90)

        output_path = output_dirpath / (input_path.stem + '.html')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(panorama_html, encoding='utf-8')

    # %%

    post_to_pipeline_server((f'{working_subdir} / {output_dirpath.name}', None))
