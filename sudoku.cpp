#include "sudoku.hpp"
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

Mat preprocess(const Mat &img);
tuple<vector<Point>, double> biggest_contours(vector<vector<Point>> contours);
vector<Point> reorder(vector<Point> points);
vector<Mat> split_boxes(Mat img);

inline string find_sudoku(const string filename) {
    const Mat img = imread(filename);
    const Mat threshold = preprocess(img);

    vector<vector<Point>> contours;
    findContours(threshold, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    auto [biggest, maxArea] = biggest_contours(contours);

    if(biggest.empty()) return "_No valid area found";

    biggest = reorder(biggest);
    const vector objective{Point(0, 0), Point(450, 0), Point(0, 450), Point(450, 450)};
    const Mat matrix = getPerspectiveTransform(biggest, objective);
    Mat warp;
    warpPerspective(img, warp, matrix, Size(450, 450));

    vector<Mat> boxes = split_boxes(warp);
    //TODO load model + predict
}

Mat preprocess(const Mat &img) {
    Mat gray, blur, threshold;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blur, Size(3, 3), 1);
    adaptiveThreshold(blur, threshold, 255, 1, 1, 11, 2);

    return threshold;
}

tuple<vector<Point>, double> biggest_contours(vector<vector<Point>> contours) {
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

vector<Point> reorder(vector<Point> points) {
    return points; //TODO
}

vector<Mat> split_boxes(Mat img) {
    vector<Mat> result;
    return result; //TODO
}
