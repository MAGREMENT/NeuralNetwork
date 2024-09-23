#include "mnist.hpp"

void print_iteration(neural_network* network, test_data* data, const int i) {
    printf("Iteration #%d done !", i + 1);
}

int main(int argc, char *argv[]) {
    if(argc != 5) {
        cout << "Expected 4 arguments : the image file and the label file for both training and testing";
        return EXIT_FAILURE;
    }

    cout << "Starting up...\n";
    vector<Mat> trainImages = read_images(argv[1]);
    vector<int> trainLabels = read_labels(argv[2]);
    test_data* trainData = to_test_data(trainImages, trainLabels);

    vector<Mat> testImages = read_images(argv[3]);
    vector<int> testLabels = read_labels(argv[4]);
    test_data* testData = to_test_data(testImages, testLabels);

    constexpr int layers[] = {784, 100, 10};
    neural_network* network = alloc_network(3, layers);
    params p;
    p.activationType = SIGMOID;
    p.costType = MEAN_SQUARED;
    p.learningRate = 1;
    apply_params(network, p);
    cout << "App started\n";

    char command = '\0';
    while(command != 'e') {
        cout << "\nSelect your command : \n"
                "e : End\n"
                "i : Show image\n"
                "c : Check\n"
                "t : Train\n"
                "s : Save\n"
                "l : Load\n";

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
            case 'c' : {
                const test_result result = test_network(network, testData);
                printf("accuracy : %.2f cost : %.5f", result.accuracy, result.cost);

                break;
            }
            case 't' : {
                int count, batchSize;
                cout << "Iterations : \n";
                cin >> count;
                cout << "Batch size : \n";
                cin >> batchSize;

                multi_learn(network, trainData, batchSize, count, print_iteration);
            }
            case 's': {
                string file;
                cout << "File : \n";
                cin >> file;

                save(network, &p, file.data());
            }
            case 'l': {
                string file;
                cout << "File : \n";
                cin >> file;

                free_network(network);
                network = initialize(file.data(), &p);
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
            const unsigned char v = mat.at<unsigned char>(r, c);
            result->values[r * mat.cols + c] = static_cast<double>(v) / 255.0;
        }
    }

    return result;
}

input_data* to_expected(const int label, const int max) {
    input_data* result = alloc_input_data(max + 1);
    for(int i = 0; i < max; i++) {
        result->values[i] = i == label ? 1 : 0;
    }

    return result;
}

test_data* to_test_data(const vector<Mat> &mats, const vector<int> &labels) {
    test_data* result = alloc_test_data(mats.size());

    for(int i = 0; i < mats.size(); i++) {
        result->inputs[i] = *to_input(mats[i]);
        result->expected[i] = *to_expected(labels[i], 9);
    }

    return result;
}
