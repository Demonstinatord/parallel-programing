#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>
#include <string>
#include <sstream>
#include <iomanip>
#include <locale.h>

using namespace std;
using namespace chrono;


string intToString(int n) {
    stringstream ss;
    ss << n;
    return ss.str();
}

vector<vector<double>> generateMatrix(int n) {
    vector<vector<double>> matrix(n, vector<double>(n));
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(0.0, 100.0);

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            matrix[i][j] = dist(gen);

    return matrix;
}

void saveMatrix(const string& filename, const vector<vector<double>>& matrix) {
    ofstream file(filename);
    int n = matrix.size();
    file << n << endl;
    file << fixed << setprecision(15);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            file << matrix[i][j];
            if (j < n - 1) file << " ";
        }
        file << endl;
    }
    file.close();
}

vector<vector<double>> multiply(const vector<vector<double>>& A,
    const vector<vector<double>>& B) {
    int n = A.size();
    vector<vector<double>> C(n, vector<double>(n, 0.0));

    for (int i = 0; i < n; ++i)
        for (int k = 0; k < n; ++k)
            for (int j = 0; j < n; ++j)
                C[i][j] += A[i][k] * B[k][j];

    return C;
}

int main() {
    
    int sizes[] = { 200, 400, 800, 1200, 1600, 2000 };
    int numSizes = 6;

    ofstream resultsFile("results.csv");
    resultsFile << "Size,Time_seconds" << endl;

    cout << "========================================" << endl;
    cout << "Matrix multiply" << endl;
    cout << "========================================" << endl;

    for (int i = 0; i < numSizes; ++i) {
        int n = sizes[i];

        cout << "\n========== Size" << n << "x" << n << " ==========" << endl;

        string folder = "data_" + intToString(n);
        string cmd = "mkdir " + folder + " 2>nul";
        system(cmd.c_str());

        cout << "Generate matrixes..." << endl;
        auto A = generateMatrix(n);
        auto B = generateMatrix(n);

        saveMatrix(folder + "/A.txt", A);
        saveMatrix(folder + "/B.txt", B);

        cout << "Multiplying..." << endl;
        auto start = high_resolution_clock::now();
        auto C = multiply(A, B);
        auto end = high_resolution_clock::now();

        double seconds = duration_cast<milliseconds>(end - start).count() / 1000.0;

        saveMatrix(folder + "/C.txt", C);

        resultsFile << n << "," << seconds << endl;

        cout << "Time: " << seconds << "sec" << endl;
        cout << "Saved in folder: " << folder << endl;
    }

    resultsFile.close();

    cout << "\n========================================" << endl;
    cout << "Results of multiply: results.csv" << endl;
    cout << "Verificate results with: python verify.py" << endl;
    cout << "========================================" << endl;

    system("pause");
    return 0;
}