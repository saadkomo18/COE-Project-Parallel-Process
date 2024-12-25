# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -O2 -mavx -Wall

# Target executable
TARGETS = black_scholes_scalar black_scholes_simd black_scholes_mimd black_scholes mmult-scalar mmult-simd mmult-mimd

# Source files
SRC = Black-Scholes-scalar.cpp Black-Scholes-simd.cpp Black-Scholes-mimd.cpp Black-Scholes.cpp mmult-scalar.cpp mmult-simd.cpp mmult-mimd.cpp

# Default target
all: $(TARGETS)

# Rule for each executable
black_scholes_scalar: Black-Scholes-scalar.cpp
	$(CXX) $(CXXFLAGS) -o black_scholes_scalar Black-Scholes-scalar.cpp

black_scholes_simd: Black-Scholes-simd.cpp
	$(CXX) $(CXXFLAGS) -o black_scholes_simd Black-Scholes-simd.cpp

black_scholes_mimd: Black-Scholes-mimd.cpp
	$(CXX) $(CXXFLAGS) -o black_scholes_mimd Black-Scholes-mimd.cpp

black_scholes: Black-Scholes.cpp
	$(CXX) $(CXXFLAGS) -o black_scholes Black-Scholes.cpp

mmult_scalar: mmult-scalar.cpp
	$(CXX) $(CXXFLAGS) -o mmult-scalr mmult-scalar.cpp

mmult_simd: mmult.simd.cpp
	$(CXX) $(CXXFLAGS) -o mmult-simd mmult-simd.cpp
mmult-mimd: mmult-mimd.cpp
	$(CXX) $(CXXFLAGS) -o mmult-mimd mmult-mimd.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Clean up build files
clean:
	rm -f $(TARGETS)
