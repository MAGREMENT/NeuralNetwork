#include "mnist.hpp"

int main() {
    read_images("C:/Users/Zach/Desktop/Perso/NeuralNetwork/mnist-data/train-images.idx3-ubyte");

    return EXIT_SUCCESS;
}

int flip(int n) {
    int result;

    const uint8_t *n1 = reinterpret_cast<uint8_t *>(&n);
    auto *n2 = reinterpret_cast<uint8_t *>(&result);

    n2[0] = n1[3];
    n2[1] = n1[2];
    n2[2] = n1[1];
    n2[3] = n1[0];

    return result;
}

inline void read_images(const char* filename) {
    FILE* fptr = fopen(filename, "r");

    int header[4];
    fread(header, sizeof(int), 4, fptr);

    int count = flip(header[1]);
    int rows = flip(header[2]);
    int cols = flip(header[3]);

    vector<Mat> result(count);
    for(int i = 0; i < count; i++) {
        Mat current(rows, cols, CV_8UC1);
    }

    //return result;
}
