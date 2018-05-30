import cv2
import numpy as np

import pyrealsense2 as rs

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
cfg = pipeline.start(config)
dev = cfg.get_device()
depth_sensor = dev.first_depth_sensor()
depth_sensor.set_option(rs.option.visual_preset, 4)

iteration = 0
preset = 0
preset_name = ''

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        iteration = iteration + 1
        if iteration > 100:
            preset = preset + 1
            iteration = 0
            range = depth_sensor.get_option_range(rs.option.visual_preset)
            preset = preset % range.max
            depth_sensor.set_option(rs.option.visual_preset, preset)
            preset_name = depth_sensor.get_option_value_description(rs.option.visual_preset, preset)
        
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, None, 0.5, 0), cv2.COLORMAP_JET)

        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(images, preset_name,(60,80), font, 4,(255,255,255),2,cv2.LINE_AA)

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()
