#include "sudoku.hpp"

#include "mnist.hpp"
#include "neural_network.h"
#include "repository.h"
#include "utils.h"
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

Mat preprocess(const Mat &img);
tuple<vector<Point>, double> biggest_contours(const vector<vector<Point>>& contours);
void reorder(vector<Point> points);
vector<Mat> split_boxes(Mat img);
char predict(neural_network* network, const Mat &img);

inline string find_sudoku(const string &filename) {
    const Mat img = imread(filename);
    const Mat threshold = preprocess(img);

    vector<vector<Point>> contours;
    findContours(threshold, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    auto [biggest, maxArea] = biggest_contours(contours);

    if(biggest.empty()) return "_No valid area found";

    reorder(biggest);
    const vector objective{Point(0, 0), Point(450, 0), Point(0, 450), Point(450, 450)};
    const Mat matrix = getPerspectiveTransform(biggest, objective);
    Mat warp;
    warpPerspective(img, warp, matrix, Size(450, 450));

    vector<Mat> boxes = split_boxes(warp);
    neural_network* network = initialize("");

    string result;
    for(const auto& box : boxes) {
        result += predict(network, box);
    }

    return result;
}

Mat preprocess(const Mat &img) {
    Mat gray, blur, threshold;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blur, Size(3, 3), 1);
    adaptiveThreshold(blur, threshold, 255, 1, 1, 11, 2);

    return threshold;
}

tuple<vector<Point>, double> biggest_contours(const vector<vector<Point>>& contours) {
    vector<Point> biggest;
    double maxArea = 0;

    for (const auto& contour : contours) {
        const double area = contourArea(contour);
        if(area < 30 || area < maxArea) continue; //TODO adapt to image size

        const double peri = arcLength(contour, true);
        vector<Point> approx;
        approxPolyDP(contour, approx, 0.02 * peri, true);

        if(approx.size() != 4) continue;

        biggest = approx;
        maxArea = area;
    }

    return {biggest, maxArea};
}

void lower_first(Point arr[]) {
    if(arr[0].y > arr[1].y) {
        const Point buffer = arr[0];
        arr[0] = arr[1];
        arr[1] = buffer;
    }
}

void reorder(vector<Point> points) {
    Point left[2];
    Point right[2];

    int lCursor = 0;
    int rCursor = 0;

    for(int i = 0; i < 4; i++) {
        int l = 0;
        int r = 0;
        const Point& curr = points.at(i);

        for(int j = i + 1; j < 4; j++) {
            if(curr.x < points.at(j).x) l++;
            else r++;
        }

        if(l > r) left[lCursor++] = curr;
        else if(r > l) right[rCursor++] = curr;
        else if(lCursor > rCursor) right[rCursor++] = curr;
        else left[lCursor++] = curr;
    }

    lower_first(left);
    lower_first(right);

    points[0] = left[0];
    points[1] = right[0];
    points[2] = left[1];
    points[3] = right[1];
}

vector<Mat> split_boxes(Mat img) {
    vector<Mat> result;
    return result; //TODO
}

bool isEmpty(Mat img) {
    for(int r = 0; r < img.rows; r++) {
        for(int c = 0; c < img.cols; c++) {
            if(img.at<unsigned char>(r, c) < 250) return false;
        }
    }

    return true;
}

char predict(neural_network* network, const Mat &img) {
    if(isEmpty(img)) return '0';

    //TODO resize to 28, 28
    input_data* data = to_input(img);
    input_data* prediction = predict(network, data);
    char v = static_cast<char>('0' + max_index(prediction->values, prediction->count));

    free_input_data(data);
    free_input_data(prediction);

    return v;
}
