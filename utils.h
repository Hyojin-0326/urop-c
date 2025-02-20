#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>

using namespace Eigen;
using namespace std;

#define STR_EXPAND(tok) #tok
#define STR(tok) STR_EXPAND(tok)

// --- 파라미터 구조체 ---
struct preprocessParams {
    float threshold = 0.9f;  // Ground Plane 제거 임계값
    int k = 10;              // KNN에서 사용할 이웃 개수
    float voxelSize = 0.05f; // Voxel 크기
};

// --- 데이터 로딩 설정 ---


struct LoaderParams{
    int width = 640;
    int height = 480;
    int num_frames;
    // LoaderParams(int frames): width(w), height(h), num_frames(frames){}
    };


// ---포인트 구조체 ---
struct Point {
    Vector3f position;  
    Vector3f normal;    
    Vector3f color; 
    bool isGround;   
    Point(const Eigen::Vector3f& pos, const Eigen::Vector3f& norm, const Eigen::Vector3f& col, bool ground)
    : position(pos), normal(norm), color(col), isGround(ground) {}  
};

// --- VoxelGrid 클래스 (공간 분할 & KNN 검색) ---
class VoxelGrid {
public:
    VoxelGrid(const std::vector<Point>& points, const preprocessParams& params);
    VoxelGrid(const std::vector<Point>& points, float voxelSize); 

    // 반경 내 검색 (radius search)
    std::vector<int> radiusSearch(const Vector3f& query, float radius,
                                  const std::vector<Point>& points) const;

    // k-NN 검색
    std::vector<int> getKNN(const Vector3f& query, int k, float searchRadius,
                            const std::vector<Point>& points) const;

private:
    float voxelSize;
    std::unordered_map<size_t, std::vector<int>> grid;
};

// --- 데이터 로딩 클래스 ---
class DataLoader {
private:


public:
    std::string data_dir;
    std::string rgb_data;
    std::string depth_data;
    std::string meta_file;
    
    DataLoader()
        : data_dir(getCurrentDir() + "/home/hjkwon/urop-c/data"),
          rgb_data(data_dir + "/rgb_data.bin"),
          depth_data(data_dir + "/depth_data.bin"),
          meta_file(data_dir + "/meta.txt") {}


    

    string getCurrentDir() {
        string fullPath = STR(__FILE__);
        size_t lastSlash = fullPath.find_last_of("/\\");
        return (lastSlash == std::string::npos) ? "." : fullPath.substr(0, lastSlash);
    }

    int getFramenum() {
        int num_frames = 0;
        ifstream meta(meta_file);
        if (meta.is_open()) {
            meta >> num_frames;
            meta.close();
            return num_frames;
        } else {
            std::cerr << "메타데이터 파일을 읽을 수 없음." << endl;
            return 1;
        }
    }

    int RGBLoader(const LoaderParams& param, vector<unsigned char>& rgb_buffer);
    int DepthLoader(const LoaderParams& param, vector<float>& depth_buffer);
    void PlayVideo(const LoaderParams& param);
    vector<uint8_t> rgb_buffer;
    vector<uint8_t> depth_buffer;
};

// --- 포인트 클라우드 관련 함수 ---
std::vector<Point> generatePointCloud(const std::vector<uint8_t>& rgb, const std::vector<float>& depth, const LoaderParams& param);
void preprocessPointCloud(std::vector<Point>& points, float voxelSize, int k, float threshold);

// --- 결과 시각화 ---
void visualizePointCloud(const std::vector<Point>& points, const LoaderParams& param);

#endif // UTILS_H




