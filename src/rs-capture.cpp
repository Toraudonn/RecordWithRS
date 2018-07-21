// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.
#include <stdio.h>
#include <iostream>
#include <ctime>
#include <boost/filesystem.hpp>
#include <boost/date_time.hpp>
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <opencv2/opencv.hpp>  // Include OpenCV2
#include "../include/cv-helpers.hpp"  

using namespace std;
using namespace cv;
using namespace rs2;
namespace fs = boost::filesystem;
namespace px = boost::posix_time;

const int HEIGHT = 720;
const int WIDTH = 1280;

// format posix_time to string
// format: %Y%m%d_%H%M
string format_time(px::ptime &now)
{
    static locale loc(
        wcout.getloc(),
        new px::wtime_facet(L"%Y%m%d_%H%M")
    );
    wstringstream wss;
    wss.imbue(loc);
    wss << now;

    // conversion to string
    wstring ws = wss.str();
    return string(ws.begin(), ws.end());
}

// creates a new directory inside of the base_dir_path
// returns the path's string
string create_dir(string &base_dir_path, string &new_dir_name)
{
    // convert std::string to fs::path
    fs::path root_dir(base_dir_path);
    fs::path new_dir(new_dir_name);
    // concat two directories
    fs::path dst_dir = root_dir / new_dir;

    // check if the directory exists
    if(fs::is_directory(dst_dir))
    {
        cout << "Directory [" << dst_dir << "] exists" << endl;
    }
    else
    {
        boost::system::error_code error; // boost error
        const bool result = fs::create_directory(dst_dir, error);
        if(!result || error)
        {
            cout << "Failed to create directory: " << dst_dir << endl;
        }
    }
    return dst_dir.string();
}

// Capture Example demonstrates how to
// capture depth and color video streams and render them to the screen
int main(int argc, char * argv[]) try
{
    // Define colorizer for visualization:
    colorizer color_map;

    config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, WIDTH, HEIGHT, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, WIDTH, HEIGHT, RS2_FORMAT_Z16, 30);

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

    string root_dir("/mnt/extHDD");
    string data_dir("raw_data");
    string data_path = create_dir(root_dir, data_dir);

    px::ptime now = px::second_clock::local_time(); // get current time
    string date_in_string(format_time(now));

    // create saving directory
    string result = create_dir(data_path, date_in_string);
    // create more directories
    string rgb("rgb");
    string depth("depth");
    string rgb_dir = create_dir(result, rgb);    
    string depth_dir = create_dir(result, depth);

    px::ptime mark = px::second_clock::local_time();
    int64_t mark_min(mark.time_of_day().minutes());

    int cnt = 0;
    while( 1 )
    {
        // Get current time and create a directory
        now = px::second_clock::local_time();
        if(now.time_of_day().minutes() > mark_min || now.time_of_day().minutes() == 0)
        {
            date_in_string = format_time(now);

            // create saving directory
            result = create_dir(data_path, date_in_string);
            cout << result << endl;

            // create more directories
            rgb_dir = create_dir(result, rgb);    
            depth_dir = create_dir(result, depth);

            mark_min = int64_t(now.time_of_day().minutes());
            cnt = 0;
        }



        frameset data = pipe.wait_for_frames(); // Wait for next set of frames from the camera
        auto aligned_frames = align.process(data);

        frame color = aligned_frames.get_color_frame();
        frame depth = aligned_frames.get_depth_frame();
        
        // Convert rs2 to cv::Mat
        auto im_c = frame_to_mat(color);
        auto im_d = frame_to_mat(depth);
        im_d *= 1000.0*scale; // Convert to millimeters

        // Update the window with new data
        if(cnt % 15 == 0)  {
            // for visualization
            frame depth_vis = color_map(aligned_frames.get_depth_frame()); // for visualization (comment out if needed)
            auto im_d_vis = frame_to_mat(depth_vis); 
            imshow("Depth", im_d_vis);
            imshow("RGB", im_c);
            cerr << "Frame: " << cnt << endl;
            waitKey(1);                  
        }
        stringstream rgb_path;
        stringstream depth_path;
        rgb_path << rgb_dir << "/" << to_string(cnt) << ".png";
        depth_path << depth_dir << "/" << to_string(cnt) << ".png";

        imwrite(rgb_path.str(), im_c);
        imwrite(depth_path.str(), im_d);

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
