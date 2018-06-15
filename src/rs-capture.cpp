// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.
#include <stdio.h>
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <opencv2/opencv.hpp>  // Include OpenCV2
#include "../include/cv-helpers.hpp"  

using namespace std;
using namespace cv;
using namespace rs2;

// Capture Example demonstrates how to
// capture depth and color video streams and render them to the screen
int main(int argc, char * argv[]) try
{
    // Define colorizer for visualization:
    colorizer color_map;

    config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);

    // Declare RealSense pipeline, encapsulating the actual device and sensors
    pipeline pipe;
    pipeline_profile selection = pipe.start(cfg);
    
    auto sensor = selection.get_device().first<depth_sensor>();
    auto scale =  sensor.get_depth_scale();
    
    cerr << scale << endl;

    // Camera warmup
    frameset frames;
    for( int i=0 ; i<30 ; i++ ){
       frames = pipe.wait_for_frames(); 
    }

    rs2::align align(RS2_STREAM_COLOR);

    int cnt = 0;
    while( 1 )
    {
        frameset data = pipe.wait_for_frames(); // Wait for next set of frames from the camera

        auto aligned_frames = align.process(data);

        frame depth_vis = color_map(aligned_frames.get_depth_frame());
        frame depth = aligned_frames.get_depth_frame();
        frame color = aligned_frames.get_color_frame();

        // Convert rs2 to cv::Mat
        auto im_d_vis = frame_to_mat(depth_vis);
        //cerr<<"convert rs-color frame to cv::Mat"<<endl;
        auto im_c = frame_to_mat(color);
        //cerr<<"convert rs-depth frame to cv::Mat"<<endl;
        auto im_d = frame_to_mat(depth);
        im_d *= 1000.0*scale;

        // Update the window with new data
        imshow("Depth", im_d_vis);
        imshow("RGB", im_c);

        waitKey(1);

        char name_c[256], name_d[256];
        sprintf( name_c,"./data/rgb/%05d.png", cnt );
        sprintf( name_d,"./data/depth/%05d.png", cnt );

        imwrite( name_c, im_c);
        imwrite( name_d, im_d);
        cerr << "Frame: " << cnt << endl;
        cnt++;
    }

    return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
    cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << endl;
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    cerr << e.what() << endl;
    return EXIT_FAILURE;
}
