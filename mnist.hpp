#ifndef MNIST_H
#define MNIST_H

extern "C" {
#include "neural_network.h"
#include "repository.h"
}
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

vector<Mat> read_images(const char* filename);
vector<int> read_labels(const char* filename);
input_data* to_input(Mat mat);
test_data* to_test_data(const vector<Mat> &mats, const vector<int> &labels);

#endif //MNIST_H
