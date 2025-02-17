#include "utils.h"
#include <cmath>
#include <algorithm>
#include <omp.h>      // OpenMP 헤더
#include <cstdint>    // 정수형
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

using namespace Eigen; 


// -- 데이터 받아오기(임시임), 아직안만듦1!!!1!!!!!!!1-- 
int RgbLoader(){
    std::vector<unsigned char> rgb_data(num_frames * width * height * 3);
    std::ifstream rgb_stream(rgb_file, std::ios::binary);
    if (rgb_stream.is_open()) {
        rgb_stream.read(reinterpret_cast<char*>(rgb_data.data()), rgb_data.size());
        rgb_stream.close();
    } else {
        std::cerr << "RGB 데이터를 읽을 수 없습니다!" << std::endl;
        return 1;
    }
}

int DepthLoader(){

}



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
