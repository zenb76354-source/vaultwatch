CXX = g++
CXXFLAGS = -O3 -std=c++17 -pthread -fopenmp
LDFLAGS = -lsecp256k1 -lssl -lcrypto -pthread -fopenmp
TARGET = vaultwatch
OBJS = main.o

all: $(TARGET)

main.o: main.cpp targets.h
	$(CXX) $(CXXFLAGS) -c main.cpp -o main.o

$(TARGET): main.o
	$(CXX) $(CXXFLAGS) main.o -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET) *.o found.txt

.PHONY: all clean
