#include "utils.h"
#include <iostream>

int main() {
    LoaderParams params = {640, 480, 1}; // 이미지 크기 설정
    std::vector<uint8_t> rgb_buffer;
    std::vector<float> depth_buffer;

    
    DataLoader loader;
    std::cout << "RGB 파일 경로: " << loader.rgb_data << std::endl;
    if (loader.RGBLoader(params, rgb_buffer) == -1 || loader.DepthLoader(params, depth_buffer) == -1) {
        return -1;
    }

    std::vector<Point> points = generatePointCloud(rgb_buffer, depth_buffer, params);
    preprocessPointCloud(points, 0.05f, 10, 0.9f);
    visualizePointCloud(points, params);

    return 0;
}
