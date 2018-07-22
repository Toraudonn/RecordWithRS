import matplotlib.pyplot as plt
import numpy as np
from pose_getter import Open3D_Chain

if __name__ == "__main__":
    chain = Open3D_Chain()
    chain.change_image("static_data/rgb.png", "static_data/depth.png")

    chain.create_rgbd()
    plt.subplot(1, 2, 1)
    plt.title('Grayscale image')
    plt.imshow(chain.get_rgb(True))
    plt.subplot(1, 2, 2)
    plt.title('Depth image')
    plt.imshow(chain.get_depths(True))
    plt.show()
    #chain.save_pcd()
    chain.compare_with_room()
