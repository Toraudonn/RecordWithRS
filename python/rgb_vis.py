import open3d as o3
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    print("Read Redwood dataset")
    color_raw = o3.read_image("../data/rgb/00060.png")
    depth_raw = o3.read_image("../data/depth/00060.png")
    d_img = np.asarray(depth_raw)
    print(d_img)
    print(d_img.dtype)
    rgbd_image = o3.create_rgbd_image_from_color_and_depth(
        color_raw, depth_raw)
    print(rgbd_image)
    plt.subplot(1, 2, 1)
    plt.title('Redwood grayscale image')
    plt.imshow(rgbd_image.color)
    plt.subplot(1, 2, 2)
    plt.title('Redwood depth image')
    plt.imshow(rgbd_image.depth)
    plt.show()



    pcd = o3.create_point_cloud_from_rgbd_image(rgbd_image, o3.PinholeCameraIntrinsic(
            o3.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    print(pcd)
    o3.draw_geometries([pcd])
    o3.write_point_cloud( 'pointcloud.pcd', pcd )
    o3.write_point_cloud( 'pointcloud.ply', pcd )
