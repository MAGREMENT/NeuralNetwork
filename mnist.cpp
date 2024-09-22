#include "mnist.hpp"

int main() {
    const auto images = read_images("C:/Users/Zach/Desktop/Perso/NeuralNetwork/mnist-data/train-images.idx3-ubyte");
    imshow("Test", images.at(0));
    waitKey(0);

    const auto labels = read_labels("C:/Users/Zach/Desktop/Perso/NeuralNetwork/mnist-data/train-labels.idx1-ubyte");
    for(int i = 0; i < 20; i++) {
        printf("%d\n", labels.at(i));
    }

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

inline vector<Mat> read_images(const char* filename) {
    FILE* fptr = fopen(filename, "rb");

    int header[4];
    fread(header, sizeof(int), 4, fptr);

    const int count = flip(header[1]);
    const int rows = flip(header[2]);
    const int cols = flip(header[3]);
    const int total = rows * cols;

    vector<Mat> result(count);
    for(int i = 0; i < count; i++) {
        const Mat current(Size(rows, cols), CV_8UC1);
        fread(current.data, sizeof(unsigned char), total, fptr);
        result.at(i) = current;
    }

    return result;
}

inline vector<int> read_labels(const char* filename) {
    FILE* fptr = fopen(filename, "rb");

    int header[2];
    fread(header, sizeof(int), 2, fptr);

    const int count = flip(header[1]);

    vector<int> result(count);
    for(int i = 0; i < count; i++) {
        unsigned char v[1];
        fread(v, sizeof(unsigned char), 1, fptr);
        result.at(i) = static_cast<int>(v[0]);
    }

    return result;
}

input_data* to_input(Mat mat) {
    const input_data* result = alloc_input_data(mat.rows * mat.cols);
    for(int r = 0; r < mat.rows; r++) {
        for(int c = 0; c < mat.cols; c++) {
            unsigned char v = mat.at<unsigned char>(r, c);
            result->values[r * mat.cols + c] = static_cast<double>(v) / 255.0;
        }
    }

    return result;
}
