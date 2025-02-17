#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>

#define STR_EXPAND(tok) #tok
#define STR(tok) STR_EXPAND(tok)

// 현재 파일(utils.cpp)의 절대 경로를 가져와서, 디렉토리 부분만 추출
std::string getCurrentDir() {
    std::string fullPath = STR(__FILE__);
    size_t lastSlash = fullPath.find_last_of("/\\");
    return (lastSlash == std::string::npos) ? "." : fullPath.substr(0, lastSlash);
}

int main() {
    // utils.cpp가 있는 디렉토리 기준으로 데이터 폴더 설정
    std::string data_dir = getCurrentDir() + "/../urop/data";  
    std::string rgb_file = data_dir + "/rgb_data.bin";
    std::string depth_file = data_dir + "/depth_data.bin";
    std::string meta_file = data_dir + "/meta.txt";

    // 프레임 개수 읽기
    int num_frames = 0;
    std::ifstream meta(meta_file);
    if (meta.is_open()) {
        meta >> num_frames;
        meta.close();
    } else {
        std::cerr << "메타데이터 파일을 읽을 수 없습니다!" << std::endl;
        return 1;
    }

    int width = 640;
    int height = 480;

    // RGB 데이터 로드 (uint8)
    std::vector<unsigned char> rgb_data(num_frames * width * height * 3);
    std::ifstream rgb_stream(rgb_file, std::ios::binary);
    if (rgb_stream.is_open()) {
        rgb_stream.read(reinterpret_cast<char*>(rgb_data.data()), rgb_data.size());
        rgb_stream.close();
    } else {
        std::cerr << "RGB 데이터를 읽을 수 없습니다!" << std::endl;
        return 1;
    }

    // Depth 데이터 로드 (float32)
    std::vector<float> depth_data(num_frames * width * height);
    std::ifstream depth_stream(depth_file, std::ios::binary);
    if (depth_stream.is_open()) {
        depth_stream.read(reinterpret_cast<char*>(depth_data.data()), depth_data.size() * sizeof(float));
        depth_stream.close();
    } else {
        std::cerr << "Depth 데이터를 읽을 수 없습니다!" << std::endl;
        return 1;
    }

    // 프레임 하나씩 OpenCV로 표시
    for (int i = 0; i < num_frames; i++) {
        cv::Mat rgb_image(height, width, CV_8UC3, &rgb_data[i * width * height * 3]);
        cv::Mat depth_image(height, width, CV_32FC1, &depth_data[i * width * height]);

        // Depth 맵을 8비트로 변환 (시각화)
        cv::Mat depth_vis;
        depth_image.convertTo(depth_vis, CV_8UC1, 255.0 / 1000.0);
        cv::applyColorMap(depth_vis, depth_vis, cv::COLORMAP_JET);

        // 표시
        cv::imshow("RGB Video", rgb_image);
        cv::imshow("Depth Video", depth_vis);

        if (cv::waitKey(30) == 'q') {
            break;
        }
    }

    cv::destroyAllWindows();
    return 0;
}
