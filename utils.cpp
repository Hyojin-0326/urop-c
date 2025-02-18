#include "utils.h"
#include <cmath>
#include <algorithm>
#include <omp.h>      // OpenMP 헤더
#include <cstdint>    // 정수형
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

using namespace Eigen; 




// // -- 데이터 받아오기(임시임) -- 
int DataLoader::RGBLoader(const LoaderParams& param, vector<unsigned char>& rgb_buffer) {
    int num_frames = param.num_frames;
    int width = param.width;
    int height = param.height;

    // 디버깅용: RGB 파일 경로 출력
    std::cout << "🔍 RGB 파일 경로: " << rgb_data << std::endl;

    // 파일 존재 여부 확인
    std::ifstream test_file(rgb_data);
    if (!test_file) {
        std::cerr << "❌ 파일이 존재하지 않음: " << rgb_data << std::endl;
        return -1;
    }
    test_file.close();

    // 파일 크기 확인
    std::ifstream rgb_stream(rgb_data, std::ios::binary | std::ios::ate);
    if (!rgb_stream.is_open()) {
        std::cerr << "❌ RGB 데이터를 읽을 수 없습니다." << std::endl;
        return -1;
    }

    std::streamsize file_size = rgb_stream.tellg();
    rgb_stream.seekg(0, std::ios::beg);

    std::cout << "📏 RGB 파일 크기: " << file_size << " bytes" << std::endl;

    // 예상 파일 크기와 비교
    std::streamsize expected_size = num_frames * width * height * 3;
    if (file_size != expected_size) {
        std::cerr << "⚠️ 예상 파일 크기와 다름! 예상: " << expected_size
                  << " bytes, 실제: " << file_size << " bytes" << std::endl;
        return -1;
    }

    // RGB 데이터 읽기
    rgb_buffer.resize(expected_size);
    rgb_stream.read(reinterpret_cast<char*>(rgb_buffer.data()), expected_size);
    rgb_stream.close();

    std::cout << "✅ RGB 데이터 로딩 완료! 총 " << num_frames << " 프레임" << std::endl;
    return 0;
}




int DataLoader::DepthLoader(const LoaderParams& param, vector<float>& depth_buffer){
    int num_frames = param.num_frames;
    int width = param.width;
    int height = param.height;

    depth_buffer.resize(num_frames * width * height);
    std::ifstream depth_stream(depth_data, std::ios::binary);
    if (depth_stream.is_open()) {
        depth_stream.read(reinterpret_cast<char*>(depth_buffer.data()), depth_buffer.size() * sizeof(float)); 
        depth_stream.close();
    } else {
        std::cerr << "Depth 데이터를 읽을 수 없습니다." << std::endl;
        return -1;
    }  
    return 0;
}

void DataLoader::PlayVideo(const LoaderParams& param) {
    int width = param.width;
    int height = param.height;
    int num_frames = param.num_frames;

    std::vector<uint8_t> rgb_buffer;
    std::vector<float> depth_buffer;

    // 데이터 로딩 체크
    if (RGBLoader(param, rgb_buffer) == -1 || DepthLoader(param, depth_buffer) == -1) {
        std::cerr << "데이터 로딩 실패!" << std::endl;
        return;
    }

    for (int i = 0; i < num_frames; i++) {
        cv::Mat rgb_image(height, width, CV_8UC3, &rgb_buffer[i * width * height * 3]);
        cv::Mat depth_image(height, width, CV_32FC1, &depth_buffer[i * width * height]);

        cv::Mat depth_vis;
        depth_image.convertTo(depth_vis, CV_8UC1, 255.0 / 1000.0);
        cv::applyColorMap(depth_vis, depth_vis, cv::COLORMAP_JET);

        cv::imshow("RGB Video", rgb_image);
        cv::imshow("Depth Video", depth_vis);

        if (cv::waitKey(30) == 'q') {
            break;
        }
    }

    cv::destroyAllWindows();
}

// --- 포인트 클라우드 생성 ---
std::vector<Point> generatePointCloud(const std::vector<uint8_t>& rgb, const std::vector<float>& depth, const LoaderParams& param) {
    std::vector<Point> points;
    
    for (int y = 0; y < param.height; y++) {
        for (int x = 0; x < param.width; x++) {
            int idx = y * param.width + x;
            float d = depth[idx]; 
            if (d <= 0 || d > 5.0f) continue;

            float X = (x - param.width / 2) * d / 525.0f;
            float Y = (y - param.height / 2) * d / 525.0f;
            float Z = d;

            // ✅ RGB 색상 추가 (RGB 데이터는 3채널이므로 idx * 3 사용)
            int rgb_idx = idx * 3;
            float R = rgb[rgb_idx] / 255.0f;
            float G = rgb[rgb_idx + 1] / 255.0f;
            float B = rgb[rgb_idx + 2] / 255.0f;

            points.push_back({Eigen::Vector3f(X, Y, Z), Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(R, G, B), false});
        }
    }
    return points;
}


// --- VoxelGrid 기반 공간 분할 ---
VoxelGrid::VoxelGrid(const std::vector<Point>& points, float voxelSize) : voxelSize(voxelSize) {
    for (size_t i = 0; i < points.size(); i++) {
        int ix = static_cast<int>(std::floor(points[i].position.x() / voxelSize));
        int iy = static_cast<int>(std::floor(points[i].position.y() / voxelSize));
        int iz = static_cast<int>(std::floor(points[i].position.z() / voxelSize));
        size_t key = (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791); // 해시 함수
        grid[key].push_back(static_cast<int>(i));
    }
}

// --- Ground Plane 제거 ---
void preprocessPointCloud(std::vector<Point>& points, float voxelSize, int k, float threshold) {
    VoxelGrid grid(points, voxelSize);
    float searchRadius = voxelSize * 1.5f;

    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(points.size()); i++) {
        Point& pt = points[i];
        std::vector<int> neighbors = grid.getKNN(pt.position, k, searchRadius, points);
        if (neighbors.size() < static_cast<size_t>(k)) {
            pt.isGround = false;
            continue;
        }

        Matrix3f covariance = Matrix3f::Zero();
        Vector3f mean = Vector3f::Zero();
        for (int idx : neighbors) mean += points[idx].position;
        mean /= neighbors.size();

        for (int idx : neighbors) {
            Vector3f diff = points[idx].position - mean;
            covariance += diff * diff.transpose();
        }
        covariance /= neighbors.size();

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
        Vector3f normal = solver.eigenvectors().col(0);

        if (normal.dot(Vector3f(0, 0, 1)) < 0) normal = -normal;
        pt.normal = normal;
        pt.isGround = (normal.dot(Vector3f(0, 0, 1)) > threshold);
    }

    points.erase(std::remove_if(points.begin(), points.end(), [](const Point& p) { return p.isGround; }), points.end());
}

// --- 시각화 ---
void visualizePointCloud(const std::vector<Point>& points, const LoaderParams& param) {
    cv::Mat display(param.height, param.width, CV_8UC3, cv::Scalar(0, 0, 0));
    for (const auto& p : points) {
        int x = static_cast<int>(p.position.x() * 100 + param.width / 2);
        int y = static_cast<int>(p.position.y() * 100 + param.height / 2);
        if (x >= 0 && x < param.width && y >= 0 && y < param.height) {
            display.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);
        }
    }
    cv::imshow("PointCloud", display);
    cv::waitKey(0);
    cv::destroyAllWindows();
}
// ///////////////////


/////////////////////////////////////////////////////////////////
// --- 내부 헬퍼 함수 ---
// voxel 좌표 (ix, iy, iz)를 하나의 key로 변환 (간단한 해시 함수)
namespace {
    inline size_t computeVoxelKey(int ix, int iy, int iz) {
        size_t h = 1469598103934665603UL;
        h = (h ^ static_cast<size_t>(ix)) * 1099511628211UL;
        h = (h ^ static_cast<size_t>(iy)) * 1099511628211UL;
        h = (h ^ static_cast<size_t>(iz)) * 1099511628211UL;
        return h;
    }
}

// --- VoxelGrid 클래스 구현 ---

VoxelGrid::VoxelGrid(const std::vector<Point>& points, const preprocessParams& params)
    : voxelSize(params.voxelSize)
{
    // 모든 포인트에 대해 voxel 좌표 계산 후 grid에 추가
    for (size_t i = 0; i < points.size(); i++) {
        int ix = static_cast<int>(std::floor(points[i].position.x() / voxelSize));
        int iy = static_cast<int>(std::floor(points[i].position.y() / voxelSize));
        int iz = static_cast<int>(std::floor(points[i].position.z() / voxelSize));
        size_t key = computeVoxelKey(ix, iy, iz);
        grid[key].push_back(static_cast<int>(i));
    }
}

std::vector<int> VoxelGrid::radiusSearch(const Vector3f& query, float radius,
                                         const std::vector<Point>& points) const
{
    std::vector<int> result;
    // 검색 반경에 해당하는 voxel 범위 계산
    int min_ix = static_cast<int>(std::floor((query.x() - radius) / voxelSize));
    int max_ix = static_cast<int>(std::floor((query.x() + radius) / voxelSize));
    int min_iy = static_cast<int>(std::floor((query.y() - radius) / voxelSize));
    int max_iy = static_cast<int>(std::floor((query.y() + radius) / voxelSize));
    int min_iz = static_cast<int>(std::floor((query.z() - radius) / voxelSize));
    int max_iz = static_cast<int>(std::floor((query.z() + radius) / voxelSize));

    float radius2 = radius * radius;
    // 해당 voxel 내의 포인트들을 순회
    for (int ix = min_ix; ix <= max_ix; ix++) {
        for (int iy = min_iy; iy <= max_iy; iy++) {
            for (int iz = min_iz; iz <= max_iz; iz++) {
                size_t key = computeVoxelKey(ix, iy, iz);
                auto it = grid.find(key);
                if (it != grid.end()) {
                    for (int idx : it->second) {
                        // 거리가 반경 이내이면 결과에 추가
                        if ((points[idx].position - query).squaredNorm() <= radius2)
                            result.push_back(idx);
                    }
                }
            }
        }
    }
    return result;
}

std::vector<int> VoxelGrid::getKNN(const Vector3f& query, int k, float searchRadius,
                                   const std::vector<Point>& points) const
{
    std::vector<int> candidates = radiusSearch(query, searchRadius, points);
    // 거리에 따라 정렬
    std::sort(candidates.begin(), candidates.end(), [&](int a, int b) {
         return (points[a].position - query).squaredNorm() < (points[b].position - query).squaredNorm();
    });
    if (candidates.size() > static_cast<size_t>(k))
         candidates.resize(k);
    return candidates;
}

// --- 포인트 클라우드 전처리 ---
// 각 포인트에 대해 k-NN를 구한 후 PCA로 법선 추정, 기준 법선 (0,0,1)과의 내적이 threshold 이상이면 ground로 판단하여 제거
void preprocessPointCloud(std::vector<Point>& points, const preprocessParams& params){
    float voxelSize = params.voxelSize;
    // 1. VoxelGrid를 미리 구축 (포인트들이 연속 메모리에 있으므로 캐시 효율 좋음)
    VoxelGrid grid(points, params);
    // k-NN 검색 시 사용할 반경 (voxelSize에 기반한 경험적 값)
    float searchRadius = voxelSize * 1.5f;
    
    // 2. 각 포인트에 대해 독립적 연산이 가능하므로 병렬 처리 (OpenMP)
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(points.size()); i++) {
        Point& pt = points[i];
        // k-NN 검색 (근사 검색)
        int k = params.k;
        std::vector<int> neighbors = grid.getKNN(pt.position, k, searchRadius, points);
        if (neighbors.size() < static_cast<size_t>(k)) {
            pt.isGround = false;
            continue;
        }
        
        // 3. PCA 연산 (고정 크기 k이므로 스택 메모리 활용 가능)
        Matrix3f covariance;
        covariance.setZero();
        Vector3f mean;
        mean.setZero();
        for (int idx : neighbors)
            mean += points[idx].position;
        mean /= neighbors.size();
        for (int idx : neighbors) {
            Vector3f diff = points[idx].position - mean;
            covariance += diff * diff.transpose();
        }
        covariance /= neighbors.size();
        // Eigen의 SelfAdjointEigenSolver는 SIMD 최적화가 되어 있음 (컴파일 옵션에 따라)
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
        // 최소 고유값에 해당하는 eigenvector가 법선
        Vector3f normal = solver.eigenvectors().col(0);
        
        // 4. 법선 방향 정렬: 센서가 고정되어 있으므로 (0,0,1) 기준 벡터와 내적
        if (normal.dot(Vector3f(0, 0, 1)) < 0)
            normal = -normal;
        pt.normal = normal;

        float threshold = params.threshold;
        // 5. 기준 법선과의 내적이 threshold 이상이면 ground plane으로 간주
        pt.isGround = (normal.dot(Vector3f(0, 0, 1)) > threshold);
    }
    
    // 6. ground plane에 해당하는 포인트들을 제거 (in-place 제거)
    points.erase(
        std::remove_if(points.begin(), points.end(), [](const Point& p) { return p.isGround; }),
        points.end()
    );
}
//////////////////////////////////////////////////////////////////