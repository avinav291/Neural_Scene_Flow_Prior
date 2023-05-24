import matplotlib.pyplot as plt
import os
import numpy as np
import open3d as o3d
import argparse

from collections import namedtuple
from itertools import accumulate
from typing import Optional, Tuple
from matplotlib.ticker import AutoMinorLocator

DEFAULT_TRANSITIONS = (15, 6, 4, 11, 13, 6)

BLUE = (94/255, 129/255, 160/255)
GREEN = (163/255, 190/255, 128/255)
RED = (191/255, 97/255, 106/255)
PURPLE = (180/255, 142/255, 160/255)
OPACITY = 1.0


def show_flows(pc1, pc2, flow, exp_dir_path, sample_id, output_name="", inverse=False, camera=None):
    if type(pc1) is not np.ndarray:
        pc1 = pc1.cpu().numpy()
        pc2 = pc2.cpu().numpy()
        flow = flow.detach().cpu().numpy()

    pcd_path = f"{exp_dir_path}/pcd_vis"
    if not os.path.exists(pcd_path):
        os.makedirs(pcd_path)
    pcd_sf_path = f"{exp_dir_path}/pcd_sf_vis"
    if not os.path.exists(pcd_sf_path):
        os.makedirs(pcd_sf_path)
    
    ins_mask1, ins_mask2 = None, None
    if pc1.shape[-1]>3 and pc2.shape[-1]>3:
        ins_mask1 = pc1[:, -1:]
        ins_mask2 = pc2[:, -1:]
        pc1 = pc1[:, :3]
        pc2 = pc2[:, :3]

        pcd_ins_path = f"{exp_dir_path}/pcd_ins_vis"
        if not os.path.exists(pcd_ins_path):
            os.makedirs(pcd_ins_path)
    
    pc1_deform = pc1 + flow

    vis = o3d.visualization.Visualizer()
    vis.create_window("visualizer", 1920, 1080, 0, 0, visible=False)
    # vis.create_window("visualizer", 960, 540, 0, 0, visible=False)
    
    pc1_o3d = o3d.geometry.PointCloud()
    pc1_o3d.points = o3d.utility.Vector3dVector(pc1)
    pc1_o3d.paint_uniform_color(BLUE)
    pc1_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pc1_o3d.orient_normals_to_align_with_direction()
    vis.add_geometry(pc1_o3d)
    vis.update_geometry(pc1_o3d)

    pc2_o3d = o3d.geometry.PointCloud()
    pc2_o3d.points = o3d.utility.Vector3dVector(pc2)
    pc2_o3d.paint_uniform_color(GREEN)
    pc2_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pc2_o3d.orient_normals_to_align_with_direction()    
    vis.add_geometry(pc2_o3d)
    vis.update_geometry(pc2_o3d)

    pc1_def_o3d = o3d.geometry.PointCloud()
    pc1_def_o3d.points = o3d.utility.Vector3dVector(pc1_deform)
    pc1_def_o3d.paint_uniform_color(RED)
    pc1_def_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pc1_def_o3d.orient_normals_to_align_with_direction()
    vis.add_geometry(pc1_def_o3d)
    vis.update_geometry(pc1_def_o3d)


    render_options = vis.get_render_option()

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(f"{pcd_path}/pcd_{sample_id}_{output_name}.jpg")
    vis.destroy_window()

    # ANCHOR: new plot style
    pc1_o3d = o3d.geometry.PointCloud()
    colors_flow = flow_to_rgb(flow, background = 'dark')
    pc1_o3d.points = o3d.utility.Vector3dVector(pc1)
    pc1_o3d.colors = o3d.utility.Vector3dVector(colors_flow / 255.0)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    vis.add_geometry(axis)

    vis.add_geometry(pc1_o3d)
    vis.update_geometry(pc1_o3d)

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(f"{pcd_sf_path}/pcd_sf_{sample_id}_{output_name}.png")
    vis.destroy_window()


    #Draw Instance Mask
    if ins_mask1 is not None and ins_mask2 is not None:
        ins_mask1 = np.repeat(ins_mask1, repeats=2, axis=1)
        ins_mask2 = np.repeat(ins_mask2, repeats=2, axis=1)
        pc1_o3d = o3d.geometry.PointCloud()
        colors_ins = flow_to_rgb(ins_mask1, background = 'dark')
        pc1_o3d.points = o3d.utility.Vector3dVector(pc1)
        pc1_o3d.colors = o3d.utility.Vector3dVector(colors_ins / 255.0)

        # pc2_o3d = o3d.geometry.PointCloud()
        # colors_ins = flow_to_rgb(ins_mask2, background = 'dark')
        # pc2_o3d.points = o3d.utility.Vector3dVector(pc2)
        # pc2_o3d.colors = o3d.utility.Vector3dVector(colors_ins / 255.0)

        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        vis.add_geometry(axis)
        vis.add_geometry(pc1_o3d)
        vis.update_geometry(pc1_o3d)
        # vis.poll_events()
        # vis.add_geometry(pc2_o3d)
        # vis.update_geometry(pc2_o3d)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(f"{pcd_ins_path}/pcd_ins_{sample_id}_{output_name}.png")
        vis.destroy_window()


def make_colorwheel(transitions: tuple=DEFAULT_TRANSITIONS) -> np.ndarray:
    """Creates a colorwheel (borrowed/modified from flowpy).
    A colorwheel defines the transitions between the six primary hues:
    Red(255, 0, 0), Yellow(255, 255, 0), Green(0, 255, 0), Cyan(0, 255, 255), Blue(0, 0, 255) and Magenta(255, 0, 255).
    Args:
        transitions: Contains the length of the six transitions, based on human color perception.
    Returns:
        colorwheel: The RGB values of the transitions in the color space.
    Notes:
        For more information, see:
        https://web.archive.org/web/20051107102013/http://members.shaw.ca/quadibloc/other/colint.htm
        http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    """
    colorwheel_length = sum(transitions)
    # The red hue is repeated to make the colorwheel cyclic
    base_hues = map(
        np.array, ([255, 0, 0], [255, 255, 0], [0, 255, 0], [0, 255, 255], [0, 0, 255], [255, 0, 255], [255, 0, 0])
    )
    colorwheel = np.zeros((colorwheel_length, 3), dtype="uint8")
    hue_from = next(base_hues)
    start_index = 0
    for hue_to, end_index in zip(base_hues, accumulate(transitions)):
        transition_length = end_index - start_index
        colorwheel[start_index:end_index] = np.linspace(hue_from, hue_to, transition_length, endpoint=False)
        hue_from = hue_to
        start_index = end_index
    return colorwheel


def flow_to_rgb(
    flow: np.ndarray,
    flow_max_radius: Optional[float]=None,
    background: Optional[str]="bright",
) -> np.ndarray:
    """Creates a RGB representation of an optical flow (borrowed/modified from flowpy).
    Args:
        flow: scene flow.
            flow[..., 0] should be the x-displacement
            flow[..., 1] should be the y-displacement
            flow[..., 2] should be the z-displacement
        flow_max_radius: Set the radius that gives the maximum color intensity, useful for comparing different flows.
            Default: The normalization is based on the input flow maximum radius.
        background: States if zero-valued flow should look 'bright' or 'dark'.
    Returns: An array of RGB colors.
    """
    valid_backgrounds = ("bright", "dark")
    if background not in valid_backgrounds:
        raise ValueError(f"background should be one the following: {valid_backgrounds}, not {background}.")
    wheel = make_colorwheel()
    # For scene flow, it's reasonable to assume displacements in x and y directions only for visualization pursposes.
    complex_flow = flow[..., 0] + 1j * flow[..., 1]
    radius, angle = np.abs(complex_flow), np.angle(complex_flow)
    if flow_max_radius is None:
        flow_max_radius = np.max(radius)
    if flow_max_radius > 0:
        radius /= flow_max_radius
    ncols = len(wheel)
    # Map the angles from (-pi, pi] to [0, 2pi) to [0, ncols - 1)
    angle[angle < 0] += 2 * np.pi
    angle = angle * ((ncols - 1) / (2 * np.pi))
    # Make the wheel cyclic for interpolation
    wheel = np.vstack((wheel, wheel[0]))
    # Interpolate the hues
    (angle_fractional, angle_floor), angle_ceil = np.modf(angle), np.ceil(angle)
    angle_fractional = angle_fractional.reshape((angle_fractional.shape) + (1,))
    float_hue = (
        wheel[angle_floor.astype(np.int)] * (1 - angle_fractional) + wheel[angle_ceil.astype(np.int)] * angle_fractional
    )
    ColorizationArgs = namedtuple(
        'ColorizationArgs', ['move_hue_valid_radius', 'move_hue_oversized_radius', 'invalid_color']
    )
    def move_hue_on_V_axis(hues, factors):
        return hues * np.expand_dims(factors, -1)
    def move_hue_on_S_axis(hues, factors):
        return 255. - np.expand_dims(factors, -1) * (255. - hues)
    if background == "dark":
        parameters = ColorizationArgs(
            move_hue_on_V_axis, move_hue_on_S_axis, np.array([255, 255, 255], dtype=np.float)
        )
    else:
        parameters = ColorizationArgs(move_hue_on_S_axis, move_hue_on_V_axis, np.array([0, 0, 0], dtype=np.float))
    colors = parameters.move_hue_valid_radius(float_hue, radius)
    oversized_radius_mask = radius > 1
    colors[oversized_radius_mask] = parameters.move_hue_oversized_radius(
        float_hue[oversized_radius_mask],
        1 / radius[oversized_radius_mask]
    )
    return colors.astype(np.uint8)


def calibration_pattern(
    pixel_size: int=151,
    flow_max_radius: float=1,
    **flow_to_rgb_args
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates a calibration pattern to add as a legend to the scene flow plots.
    Args:
        pixel_size: Radius of the square test pattern.
        flow_max_radius: The maximum radius value represented by the calibration pattern.
        flow_to_rgb_args: kwargs passed to the flow_to_rgb function.
    Returns:
        calibration_img: The RGB image representation of the calibration pattern.
        calibration_flow: The flow represented in the calibration_pattern.
    """
    half_width = pixel_size // 2
    y_grid, x_grid = np.mgrid[:pixel_size, :pixel_size]
    u = flow_max_radius * (x_grid / half_width - 1)
    v = flow_max_radius * (y_grid / half_width - 1)
    flow = np.zeros((pixel_size, pixel_size, 2))
    flow[..., 0] = u
    flow[..., 1] = v
    flow_to_rgb_args["flow_max_radius"] = flow_max_radius
    img = flow_to_rgb(flow, **flow_to_rgb_args)
    return img, flow


def attach_calibration_pattern(ax, **calibration_pattern_kwargs):
    """Attach a calibration pattern to axes.
    This function uses calibration_pattern to generate a figure.
    Args:
        calibration_pattern_kwargs: kwargs, optional
            Parameters to be given to the calibration_pattern function.
    Returns:
        image_axes: matplotlib.AxesImage
            See matplotlib.imshow documentation
            Useful for changing the image dynamically
        circle_artist: matplotlib.artist
            See matplotlib.circle documentation
            Useful for removing the circle from the figure
    """
    
    pattern, flow = calibration_pattern(**calibration_pattern_kwargs)
    flow_max_radius = calibration_pattern_kwargs.get("flow_max_radius", 1)
    extent = (-flow_max_radius, flow_max_radius) * 2
    image = ax.imshow(pattern, extent=extent)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    for spine in ("bottom", "left"):
        ax.spines[spine].set_position("zero")
        ax.spines[spine].set_linewidth(1)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    attach_coord(ax, flow, extent=extent)
    circle = plt.Circle((0, 0), flow_max_radius, fill=False, lw=1)
    ax.add_artist(circle)
    
    return image, circle


def attach_coord(ax, flow, extent=None):
    """Attach the flow value to the coordinate tooltip.
    It allows you to see on the same figure, the RGB value of the pixel and the underlying value of the flow.
    Shows cartesian and polar coordinates.
    Args:
        ax: matplotlib.axes
            The axes the arrows should be plotted on.
        flow: numpy.ndarray
            scene flow.
            flow[..., 0] should be the x-displacement
            flow[..., 1] should be the y-displacement
        extent: sequence_like, optional
            Use this parameters in combination with matplotlib.imshow to resize the RGB plot.
            See matplotlib.imshow extent parameter.
            See attach_calibration_pattern
    """
    
    height, width, _ = flow.shape
    base_format = ax.format_coord
    if extent is not None:
        left, right, bottom, top = extent
        x_ratio = width / (right - left)
        y_ratio = height / (top - bottom)
        
    def new_format_coord(x, y):
        if extent is None:
            int_x = int(x + 0.5)
            int_y = int(y + 0.5)
        else:
            int_x = int((x - left) * x_ratio)
            int_y = int((y - bottom) * y_ratio)
        if 0 <= int_x < width and 0 <= int_y < height:
            format_string = "Coord: x={}, y={} / Flow: ".format(int_x, int_y)
            u, v = flow[int_y, int_x, :]
            if np.isnan(u) or np.isnan(v):
                format_string += "invalid"
            else:
                complex_flow = u - 1j * v
                r, h = np.abs(complex_flow), np.angle(complex_flow, deg=True)
                format_string += ("u={:.2f}, v={:.2f} (cartesian) ρ={:.2f}, θ={:.2f}° (polar)"
                                  .format(u, v, r, h))
            return format_string
        else:
            return base_format(x, y)
        
    ax.format_coord = new_format_coord


def custom_draw_geometry_with_key_callback(pcds):
    def change_background_to_black(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([76/255, 86/255, 106/255])
        # opt.background_color = np.asarray([7/255, 54/255, 66/255])
        return False
    
    # def load_render_option(vis):
    #     vis.get_render_option().load_from_json(
    #         "../../TestData/renderoption.json")
    #     return False
    
    def capture_depth(vis):
        depth = vis.capture_depth_float_buffer()
        plt.imshow(np.asarray(depth))
        plt.show()
        return False
    
    def capture_image(vis):
        image = vis.capture_screen_float_buffer()
        plt.imshow(np.asarray(image))
        plt.show()
        return False
    
    key_to_callback = {}
    key_to_callback[ord("K")] = change_background_to_black
    # key_to_callback[ord("R")] = load_render_option
    key_to_callback[ord(",")] = capture_depth
    key_to_callback[ord(".")] = capture_image
    o3d.visualization.draw_geometries_with_key_callbacks(pcds, key_to_callback)

def custom_draw_geometry_with_camera_trajectory(pcd, camera_trajectory_path,
                                                render_option_path,
                                                output_path):
    custom_draw_geometry_with_camera_trajectory.index = -1
    custom_draw_geometry_with_camera_trajectory.trajectory =\
        o3d.io.read_pinhole_camera_trajectory(camera_trajectory_path)
    custom_draw_geometry_with_camera_trajectory.vis = o3d.visualization.Visualizer(
    )
    image_path = os.path.join(output_path, 'image')
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    depth_path = os.path.join(output_path, 'depth')
    if not os.path.exists(depth_path):
        os.makedirs(depth_path)

    print("Saving color images in " + image_path)
    print("Saving depth images in " + depth_path)

    def move_forward(vis):
        # This function is called within the o3d.visualization.Visualizer::run() loop
        # The run loop calls the function, then re-render
        # So the sequence in this function is to:
        # 1. Capture frame
        # 2. index++, check ending criteria
        # 3. Set camera
        # 4. (Re-render)
        ctr = vis.get_view_control()
        glb = custom_draw_geometry_with_camera_trajectory
        if glb.index >= 0:
            print("Capture image {:05d}".format(glb.index))
            # Capture and save image using Open3D.
            vis.capture_depth_image(
                os.path.join(depth_path, "{:05d}.png".format(glb.index)), False)
            vis.capture_screen_image(
                os.path.join(image_path, "{:05d}.png".format(glb.index)), False)

            # Example to save image using matplotlib.
            '''
            depth = vis.capture_depth_float_buffer()
            image = vis.capture_screen_float_buffer()
            plt.imsave(os.path.join(depth_path, "{:05d}.png".format(glb.index)),
                       np.asarray(depth),
                       dpi=1)
            plt.imsave(os.path.join(image_path, "{:05d}.png".format(glb.index)),
                       np.asarray(image),
                       dpi=1)
            '''

        glb.index = glb.index + 1
        if glb.index < len(glb.trajectory.parameters):
            ctr.convert_from_pinhole_camera_parameters(
                glb.trajectory.parameters[glb.index])
        else:
            custom_draw_geometry_with_camera_trajectory.vis.destroy_window()

        # Return false as we don't need to call UpdateGeometry()
        return False

    vis = custom_draw_geometry_with_camera_trajectory.vis
    vis.create_window()
    vis.add_geometry(pcd)
    vis.register_animation_callback(move_forward)
    vis.run()  

def main(dir_path):
    outputs = sorted(os.listdir(dir_path))[1:]
    pcd_sf_path = f"{dir_path}/../best_sf_output"
    if not os.path.exists(pcd_sf_path):
        os.makedirs(pcd_sf_path, exist_ok=True)

    for idx,output in enumerate(outputs):
        path = os.path.join(dir_path, output)
        data = np.load(path)

        pc1 = data["p1"].astype('float32')[0]
        print(pc1.shape)
        # pc2 = data["pc2"].astype('float32')
        flow = data['flow'].astype('float32')[0]
        
        # ANCHOR: new plot style
        pc1_o3d = o3d.geometry.PointCloud()
        colors_flow = flow_to_rgb(flow, background = 'dark')
        pc1_o3d.points = o3d.utility.Vector3dVector(pc1)
        pc1_o3d.colors = o3d.utility.Vector3dVector(colors_flow / 255.0)
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(pc1_o3d)
        vis.update_geometry(pc1_o3d)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(f"{pcd_sf_path}/pcd_sf_{idx}.png")
        vis.destroy_window()

        # new_plot(pc1,pc2, flow)
        # print('finish:',path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dir',default='/scratch/ag7644/nsfp/checkpoints/waymo_sf_debug11_1/sceneflow_nsfp/',help='path to the waymo open dataset')
    parser.add_argument('--dir',default='/scratch/ag7644/nsfp/checkpoints/Argoverse_train_ins_3/sceneflow_nsfp',help='path to the waymo open dataset')
    args = parser.parse_args()

    main(args.dir)