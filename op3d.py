#这个文件是用于真实的realsense相机
import open3d as o3d
#o3d.t.io.RealSenseSensor.list_devices()
#列出相机信息
import numpy as np
import math
import time
import cv2
import numpy as np
import pyrealsense2 as rs
import PIL.Image as Image
# Configure depth and color streams
pc = rs.pointcloud()
points = rs.points()
#align = rs.align()
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
#open3d要求彩色图与深度图大小一致，如果配置时候分辨率不同，需要手动对齐，open3d与realsense都有对齐工具，后面采取后者的方法
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)
# 指定对齐对象
align_to = rs.stream.color
align = rs.align(align_to)
try:
    # 放掉前几帧
    for fid in range(20):
        frames = pipeline.wait_for_frames()
        # frames = align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

    t0 = time.time()

    # aligned_frames = pipeline.wait_for_frames()
    frames = pipeline.wait_for_frames()
    print('获取耗时', time.time() - t0)
    t1 = time.time()
    aligned_frames = align.process(frames)
    print('对齐耗时', time.time() - t1)
    # 这里开始将realsense的数据转换为open3d的数据结构
    # 相机参数
    profile = aligned_frames.get_profile()
    intrinsics = profile.as_video_stream_profile().get_intrinsics()
    # 转换为open3d中的相机参数
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)
    # frames = align.process(frames)
    # 转化数据帧为图像
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    cv2.imwrite('image/rgb' + str(2) + '.png', color_image)
    depImg = Image.fromarray(depth_image)
    depImg.save('image/depth' + str(2) + '.png')
    # 转化为open3d中的图像
    t2 = time.time()
    img_depth = o3d.geometry.Image(depth_image)
    #     print(type(img_depth))
    #     print(img_depth.dimension)
    img_color = o3d.geometry.Image(color_image)
    #     print(type(img_color))
    #     print(img_color.is_empty)
    # 从彩色图、深度图创建RGBD
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(img_color, img_depth)
    # 创建pcd
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)
    print('处理耗时', time.time() - t2)
    print('总耗时', time.time() - t0)
    # 图像上下翻转
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # 可视化
    o3d.visualization.draw_geometries([pcd])#,  zoom=0.3412,  front=[0, 0, 0],lookat=[0, 0, 0],p=[0, 0, 0])

finally:
    pipeline.stop()
    print('done')
