#include "utils.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <Eigen/Dense>
using namespace Eigen;

int main()
{
    // 센서로부터 포인트 클라우드 데이터를 읽어오는 부분 (여기서는 예제이므로 더미 데이터를 생성)
    std::vector<Point> points;
    const int numPoints = 10000;
    points.reserve(numPoints);
    
    // 예시: 임의의 포인트 생성 (실제 프로젝트에서는 센서 API를 통해 데이터를 획득)
    for (int i = 0; i < numPoints; i++) {
        Point pt;
        pt.position.setRandom();
        pt.isGround = false;
        points.push_back(pt);
    }
    
    preprocessParams Params;
    // 전처리: ground plane으로 간주되는 포인트 제거
    preprocessPointCloud(points, Params);
    
    std::cout << "Ground 제거 후 남은 포인트 수: " << points.size() << std::endl;
    
    // 이후 처리 루틴...
    
    return 0;
}
