#include "mnist.hpp"

int main(int argc, char *argv[]) {
    if(argc != 5) {
        cout << "Expected 4 arguments : the image file and the label file for both training and testing";
        return EXIT_FAILURE;
    }

    cout << "Starting up...\n";
    vector<Mat> trainImages = read_images(argv[1]);
    vector<int> trainLabels = read_labels(argv[2]);
    vector<Mat> testImages = read_images(argv[3]);
    vector<int> testLabels = read_labels(argv[4]);
    cout << "App started\n";

    char command = '\0';
    while(command != 'e') {
        cout << "\nSelect your command : \n"
                "e : End\n"
                "i : Show image\n";

        cin >> command;
        switch (command) {
            case 'e' :
                break;
            case 'i' : {
                cout << "Select index : \n";
                int index;
                cin >> index;
                if(index < 0 || index > trainImages.size()) {
                    cout << "Incorrect index\n";
                    break;
                }

                auto s = to_string(trainLabels[index]);
                imshow(s, trainImages[index]);
                waitKey(0);
                destroyWindow(s);

                break;
            }
            default :
                cout << "Command not recognized\n";
        }
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
    input_data* result = alloc_input_data(mat.rows * mat.cols);
    for(int r = 0; r < mat.rows; r++) {
        for(int c = 0; c < mat.cols; c++) {
            unsigned char v = mat.at<unsigned char>(r, c);
            result->values[r * mat.cols + c] = static_cast<double>(v) / 255.0;
        }
    }

    return result;
}
