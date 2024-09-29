#include "sudoku.hpp"

extern "C" {
#include "neural_network.h"
#include "repository.h"
#include "utils.h"
}
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
    if(argc != 2) {
        cout << "Expected 1 argument : the image path";
        return EXIT_FAILURE;
    }

    cout << find_sudoku(argv[1]);

    return EXIT_SUCCESS;
}

Mat preprocess(const Mat &img);
tuple<vector<Point>, double> biggest_contours(const vector<vector<Point>>& contours);
void reorder(vector<Point> points);
Mat remove_lines_static(Mat wholeImg, int lowerRow, int upperRow, int lowerColumn, int upperColumn);
vector<Mat> split_boxes(Mat img, Mat (*remove_lines)(Mat, int, int, int, int));
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

    vector<Mat> boxes = split_boxes(warp, remove_lines_static);
    neural_network* network = initialize("", nullptr);

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

Mat remove_lines_static(Mat wholeImg, int lowerRow, int upperRow, int lowerColumn, int upperColumn) {
    const int rDelta = upperRow - lowerRow;
    lowerRow += rDelta;
    upperRow -= rDelta;

    const int cDelta = upperColumn - lowerColumn;
    lowerColumn += cDelta;
    upperColumn -= cDelta;

    Mat result(Size(upperRow - lowerRow, upperColumn - lowerColumn), CV_8UC1);
    for(int r = lowerRow; r < upperRow; r++) {
        for(int c = lowerColumn; c < upperColumn; c++) {
            result.at<unsigned char>(r - lowerRow, c - lowerColumn) = wholeImg.at<unsigned char>(r, c);
        }
    }

    return result;
}

vector<Mat> split_boxes(Mat img, Mat (*remove_lines)(Mat, int, int, int, int)) {
    vector<Mat> result;
    const int rSize = img.rows / 9;
    const int cSize = img.cols / 9;

    for(int r = 0; r < img.rows; r += rSize) {
        for(int c = 0; c < img.cols; c += cSize) {
            result.push_back(remove_lines(img, r, r + rSize, c, c + cSize));
        }
    }

    return result;
}

bool isEmpty(Mat img) {
    for(int r = 0; r < img.rows; r++) {
        for(int c = 0; c < img.cols; c++) {
            if(img.at<unsigned char>(r, c) < 250) return false;
        }
    }

    return true;
}

input_data* to_input(Mat mat) {
    input_data* result = alloc_input_data(mat.rows * mat.cols);
    for(int r = 0; r < mat.rows; r++) {
        for(int c = 0; c < mat.cols; c++) {
            const unsigned char v = mat.at<unsigned char>(r, c);
            result->values[r * mat.cols + c] = static_cast<double>(v) / 255.0;
        }
    }

    return result;
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
