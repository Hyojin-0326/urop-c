#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace Eigen;
using namespace std;

//파라미터 묶음
struct preprocessParams{
    float threshold = 0.9f; 
    int k=10;
    float voxelSize=0.05f;
};

// 포인트 구조체 (포인트 위치, 법선, ground 여부 플래그)
struct Point {
    Vector3f position;
    Vector3f normal;
    bool isGround;
};


// Voxel Grid를 통한 근접점 검색 클래스
#include <unordered_map>
class VoxelGrid {
public:

    // 생성자: 포인트 클라우드와 voxel 크기를 받아 grid를 구축함
    VoxelGrid(const std::vector<Point>& points, const preprocessParams& params);

    // 반경 내 검색 (주어진 query 위치로부터 radius 이내의 포인트 인덱스 반환)
    std::vector<int> radiusSearch(const Vector3f& query, float radius,
                                  const std::vector<Point>& points) const;

    // k-NN 검색: 반경 내 후보들 중 거리순 정렬 후 k개 반환
    std::vector<int> getKNN(const Vector3f& query, int k, float searchRadius,
                            const std::vector<Point>& points) const;
private:
    float voxelSize;
    // voxel key와 해당 voxel에 포함된 포인트 인덱스 목록 매핑
    std::unordered_map<size_t, std::vector<int>> grid;
};

// 포인트 클라우드 전처리 


// - 각 포인트에 대해 k-NN 검색 후 PCA로 법선 추정  
// - 기준 법선 (여기서는 (0,0,1)로 가정)과 내적 값이 threshold 이상이면 ground로 간주하여 제거
void preprocessPointCloud(std::vector<Point>& points, const preprocessParams& params);

//비디오 데이터 로드 임시 클래스
#include <opencv2/opencv.hpp>



#define STR_EXPAND(tok) #tok
#define STR(tok) STR_EXPAND(tok)

class DataLoader{

    private:
    std::string data_dir;
    std::string rgb_data;
    std::string depth_data;
    std::string meta_file;  

    public:
    DataLoader():
        data_dir(getCurrentDir() + "/../urop/data"),
        rgb_data(data_dir + "/rgb_data.bin"),
        depth_data(data_dir + "/depth_data.bin"),
        meta_file(data_dir + "/meta.txt"){}
    
        struct DataParams{
            int num_frames
            
        };


        string getCurrentDir(){
            string fullPath = STR(__FILE__);
            size_t lastSlash = fullPath.find_last_of("/\\");
            return (lastSlash == std::string::npos) ? "." : fullPath.substr(0, lastSlash);
        }

        int getFramenum(){
            int num_frames = 0;
            ifstream meta(meta_file);
            if (meta.is_open()){
                meta >> num_frames;
                meta.close();
                return num_frames;
            }
            else{
                std::cerr<<"메타데이터 파일을 읽을 수 없음."<<endl;
                return 1;
            }


        }

};



#endif // UTILS_H
