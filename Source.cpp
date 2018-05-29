// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2015 Intel Corporation. All Rights Reserved.

///////////////////////////////////////////////////////////
// librealsense tutorial #2 - Accessing multiple streams //
///////////////////////////////////////////////////////////

// First include the librealsense C++ header file
#include <librealsense/rs.hpp>
#include <cstdio>
#include <conio.h>
#include "../../use_PCL.h"
#include "../../use_OpenCV.h"


#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

// Also include GLFW to allow for graphical display
#include <GLFW/glfw3.h>

using namespace std;

int main( int argc, char** argv ) try
{
	// Check Options
	for( int idx = 1 ; idx<argc ; idx++ ){
//		if(      !strcmp(argv[idx], "-stream") ) mode = true;
	}
	
	cerr<<"Options------------------------"<<endl;
	cerr<<" press 's': recording mode     "<<endl;
	cerr<<" press 'e': stop recording     "<<endl;
	cerr<<" press 'q': quit               "<<endl;
	cerr<<"-------------------------------"<<endl;

	// Create a context object. This object owns the handles to all connected realsense devices.
	rs::context ctx;
	printf("There are %d connected RealSense devices.\n", ctx.get_device_count());
	if (ctx.get_device_count() == 0) return EXIT_FAILURE;

	// This tutorial will access only a single device, but it is trivial to extend to multiple devices
	rs::device * dev = ctx.get_device(0);
	printf("\nUsing device 0, an %s\n", dev->get_name());
	printf("    Serial number: %s\n", dev->get_serial());
	printf("    Firmware version: %s\n", dev->get_firmware_version());

	// Configure all streams to run at VGA resolution at 60 frames per second
	dev->enable_stream(rs::stream::depth, 640, 480, rs::format::z16, 30);
	dev->enable_stream(rs::stream::color, 640, 480, rs::format::rgb8, 30);
	dev->enable_stream(rs::stream::infrared, 640, 480, rs::format::y8, 30);
	try { dev->enable_stream(rs::stream::infrared2, 640, 480, rs::format::y8, 30); }
	catch (...) { printf("Device does not provide infrared2 stream.\n"); }
	dev->start();

	cv::Mat im_depth(480, 640, CV_16UC1);
	cv::Mat im_color(480, 640, CV_8UC3);
	float scale = dev->get_depth_scale();
	std::cerr << "dev->scale: " << scale << std::endl;

	bool record = false;
	int n_record_dir = 0; //˜A‘±‚ÅŽB‚é‚Æ‚«—p‚Ì˜A”Ô

	int	cnt = 0;
	while(1){
		dev->wait_for_frames();
		const uint16_t * depth_image = (const uint16_t *)dev->get_frame_data(rs::stream::depth_aligned_to_color);
		const uint8_t * color_image = (const uint8_t *)dev->get_frame_data(rs::stream::color);
        rs::intrinsics depth_intrin = dev->get_stream_intrinsics(rs::stream::depth);
        rs::intrinsics color_intrin = dev->get_stream_intrinsics(rs::stream::color);
        rs::extrinsics depth_to_color = dev->get_extrinsics(rs::stream::depth, rs::stream::color);
		rs::intrinsics dc_intrin = dev->get_stream_intrinsics( rs::stream::depth_aligned_to_color );

		/*
		cerr<<"\n"<<endl;
		for( int i=0 ; i<5 ; i++ )
			cerr<<"coeffs["<<i<<"] "<<dc_intrin.coeffs[i]<<endl;
		cerr<<"fx: "<<dc_intrin.fx<<endl;
		cerr<<"fy: "<<dc_intrin.fy<<endl;
		cerr<<"ppy: "<<dc_intrin.ppx<<endl;
		cerr<<"ppy: "<<dc_intrin.ppy<<endl;
		cerr<<"height: "<<dc_intrin.height<<endl;
		cerr<<"width: "<<dc_intrin.width<<endl;
		*/

		/*
				printf( "rot\n ");
				for( int i=0 ; i<9 ; i++ )
				printf( "%f\n", depth_to_color.rotation[i] );
				printf( "trans\n ");
				for( int i=0 ; i<3 ; i++ )
				printf( "%f\n", depth_to_color.translation[i] );
				printf( "color intrin\n ");
				printf( "ppx: %f\n", color_intrin.ppx);
				printf( "ppy: %f\n", color_intrin.ppy);
				printf( "fx: %f\n", color_intrin.fx);
				printf( "fy: %f\n", color_intrin.fy);
				for( int i=0 ; i<5 ; i++ )
				printf( "%f\n", color_intrin.coeffs[i]);
				*/
	
		for (int i = 0; i < im_color.rows*im_color.cols; i++){
			im_color.data[(3 * i)] = color_image[(3 * i) + 2];
			im_color.data[(3 * i) + 1] = color_image[(3 * i) + 1];
			im_color.data[(3 * i) + 2] = color_image[(3 * i)];
			im_depth.at<unsigned short>(i / im_color.cols, i%im_color.cols) = depth_image[i];
		}

		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
		for( int j=0 ; j<im_color.rows ; j++ ){
			for( int i=0 ; i<im_color.cols ; i++ ){

				// Retrieve the 16-bit depth value and map it into a depth in meters
                uint16_t depth_value = depth_image[j * depth_intrin.width + i];
				float depth_in_meters = depth_value * scale;
				// Skip over pixels with a depth value of zero, which is used to indicate no data
				if(depth_value == 0) depth_value = 0.0;

				// Map from pixel coordinates in the depth image to pixel coordinates in the color image
				rs::float2 depth_pixel = {(float)i, (float)j};
				rs::float3 depth_point = depth_intrin.deproject(depth_pixel, depth_in_meters);
				pcl::PointXYZ pnt( depth_point.x, depth_point.y, depth_point.z );
				cloud->points.push_back( pnt );
			}
		}
		cloud->width = im_color.cols;
		cloud->height = im_color.rows;


		cv::imshow("im_depth", im_depth);
		cv::imshow("im_color", im_color);
		char dir_name[256];
		if( _kbhit() ){
			char buf = _getch();
			if( buf == 's' ){
				record = true;
				char command[256];
				sprintf( dir_name,"Video%03d", n_record_dir );
				sprintf( command,"mkdir %s", dir_name );
				system( command );
				cerr<<"Recording "<< dir_name <<endl;
			}else if( buf == 'e' ){
				cerr<<"# of captured frames: "<<cnt<<endl;
				record = false;
				cnt = 0;
				n_record_dir++;
				cerr<<"Capture end"<<endl;
				cerr<<"Frame reset: "<<cnt<<endl;
			}else if( buf == 'q' ){
				cerr<<"Quit"<<endl;
				break;
			}else{
				cerr << buf << " is not assigned by any functions." << endl;
			}
		}
		cv::waitKey( 1 );

		if( record ){
			char name_d[256], name_c[256], name_p[256];
			sprintf( name_d,"%s\\depth%05d.png", dir_name, cnt );
			sprintf( name_c,"%s\\image%05d.png", dir_name, cnt );
			sprintf( name_p,"%s\\cloud_%04d.pcd", dir_name, cnt );
			cv::imwrite( name_d, im_depth );
			cv::imwrite( name_c, im_color );
			pcl::io::savePCDFileBinary( name_p, *cloud );
			cerr << "frame " << cnt << " is saved" << endl;
			cnt++;
		}

	}

	return EXIT_SUCCESS;
}
catch (const rs::error & e)
{
	// Method calls against librealsense objects may throw exceptions of type rs::error
	printf("rs::error was thrown when calling %s(%s):\n", e.get_failed_function().c_str(), e.get_failed_args().c_str());
	printf("    %s\n", e.what());
	return EXIT_FAILURE;
}
