// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.
#include <stdio.h>
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <opencv2/opencv.hpp>
// #include "example.hpp"          // Include short list of convenience functions for rendering

using namespace std;
using namespace cv;

// Capture Example demonstrates how to
// capture depth and color video streams and render them to the screen
int main(int argc, char * argv[]) try
{
	
	cout <<"Options------------------------"<<endl;
	cout <<" press 's': recording mode     "<<endl;
	cout <<" press 'e': stop recording     "<<endl;
	cout <<" press 'q': quit               "<<endl;
	cout <<"-------------------------------"<<endl;

	// Create a context object. This object owns the handles to all connected realsense devices.
    rs2::context ctx;
    auto list = ctx.query_devices(); // Get a snapshot of currently connected devices
    if (list.size() == 0) 
        throw runtime_error("No device detected. Is it plugged in?");
    rs2::device dev = list.front();

    //Create a configuration for configuring the pipeline with a non default profile
    rs2::config cfg;

    //Add desired streams to configuration
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);

	// This tutorial will access only a single device, but it is trivial to extend to multiple devices
	printf("\nUsing device 0, an %s\n", dev.get_info(RS2_CAMERA_INFO_NAME));
	printf("    Serial number: %s\n", dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));

    // Declare RealSense pipeline, encapsulating the actual device and sensors
    rs2::pipeline pipe;
    //Instruct pipeline to start streaming with the requested configuration
    rs2::pipeline_profile selection = pipe.start(cfg);

	Mat im_depth(480, 640, CV_16UC1);
	Mat im_color(480, 640, CV_8UC3);

    //auto sensor = selection.get_device().first<rs2::depth_sensor>();
    //float scale =  sensor.get_depth_scale(); 
	//cerr << "dev->scale: " << scale << endl;

    rs2::colorizer color_map;

	while(1)
    {
		rs2::frameset frames = pipe.wait_for_frames(); // Wait for next set of frames from the camera

        rs2::frame depth_frame = frames.get_depth_frame(); // Find and colorize the depth data
        rs2::frame color_frame = frames.get_color_frame(); // Find the color data
        rs2::frame depth_color = color_map(frames.get_depth_frame());

        // const uint16_t * depth_image = (const uint16_t *) depth.as<rs2::video_frame>();
		// const uint8_t * color_image = (const uint8_t *) color.as<rs2::video_frame>();

        // auto depth_stream = selection.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
        // auto color_stream = selection.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();

        // auto depth_resolution = std::make_pair(depth_stream.width(), depth_stream.height());
        // auto color_resolution = std::make_pair(color_stream.width(), depth_stream.height());
        
        // cout << "w: " << depth_resolution.width() << ", " << "h: " << depth_resolution.height() << endl;
	
		// for (int i = 0; i < im_color.rows*im_color.cols; i++)
        // {
		// 	im_color.data[(3 * i)] = color_image[(3 * i) + 2];
		// 	im_color.data[(3 * i) + 1] = color_image[(3 * i) + 1];
		// 	im_color.data[(3 * i) + 2] = color_image[(3 * i)];
		// 	im_depth.at<unsigned short>(i / im_color.cols, i%im_color.cols) = depth_image[i];
		// }

        Mat depth_show(Size(640, 480), CV_16UC1, (void*)depth_frame.get_data());
        depth_show *= 1.0;
        Mat color_show(Size(640, 480), CV_8UC3, (void*)color_frame.get_data());
        Mat depth_color_show(Size(640, 480), CV_8UC3, (void*)depth_color.get_data());

        Mat normalized_depth;
        normalize(depth_show, normalized_depth, 255, 0, NORM_MINMAX, CV_8U);

		imshow("im_depth", depth_color_show);
		imshow("im_color", color_show);

		waitKey( 1 );
	}

    return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
