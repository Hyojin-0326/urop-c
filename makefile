# 컴파일러 설정
CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++17 -I/usr/include/eigen3
LDFLAGS = -fopenmp

# 대상 파일
TARGET = urop
SRC = main.cpp utils.cpp
OBJ = $(SRC:.cpp=.o)

# 기본 빌드 규칙
all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -fopenmp -Wno-class-memaccess -o $@ $^ `pkg-config --cflags --libs opencv4`

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -fopenmp -Wno-class-memaccess -c $< -o $@ `pkg-config --cflags --libs opencv4`

# 클린업
clean:
	rm -f $(OBJ) $(TARGET)



