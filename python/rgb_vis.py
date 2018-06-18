import open3d as o3
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    print("Read Redwood dataset")
    color_raw = o3.read_image("static_data/rgb.png")
    depth_raw = o3.read_image("static_data/depth.png")
    d_img = np.asarray(depth_raw)
    print(d_img)
    print(d_img.dtype)
    rgbd_image = o3.create_rgbd_image_from_color_and_depth(color_raw, depth_raw)
    print(rgbd_image)
    # plt.subplot(1, 2, 1)
    # plt.title('Redwood grayscale image')
    # plt.imshow(rgbd_image.color)
    # plt.subplot(1, 2, 2)
    # plt.title('Redwood depth image')
    # plt.imshow(rgbd_image.depth)
    # plt.show()

    camera_intrinsic = o3.read_pinhole_camera_intrinsic("static_data/d415.json")
    P = np.loadtxt('static_data/T.csv', delimiter=',')
    
    pcd = o3.create_point_cloud_from_rgbd_image(rgbd_image, camera_intrinsic)
    # Flip it, otherwise the pointcloud will be upside down
    # pcd.transform([[1000, 0, 0, 0], [0, -1000, 0, 0], [0, 0, -1000, 0], [0, 0, 0, 1]])
    pcd.transform([[1000, 0, 0, 0], [0, -1000, 0, 0], [0, 0, -1000, 0], [0, 0, 0, 1]])
    pcd.transform(P)
    pc_room = o3.read_point_cloud('static_data/room_mode_1.ply')

    # pcd.transform()
    print(pcd)
    o3.draw_geometries([pc_room, pcd])
    #o3.write_point_cloud( 'static_data/pointcloud.pcd', pcd )
    #o3.write_point_cloud( 'static_data/pointcloud.ply', pcd )
