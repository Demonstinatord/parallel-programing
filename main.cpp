#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>
#include <string>
#include <sstream>
#include <iomanip>
#include <omp.h>
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
    #pragma omp parallel for collapse(2)
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
    const vector<vector<double>>& B, int numThreads) {
    int n = A.size();
    vector<vector<double>> C(n, vector<double>(n, 0.0));
    omp_set_num_threads(numThreads);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < n; ++k)
            for (int j = 0; j < n; ++j)
                C[i][j] += A[i][k] * B[k][j];

    return C;
}

int main() {
    
    int sizes[] = { 200, 400, 800, 1200, 1600, 2000 };
    int numSizes = 6;
    int threads[]={1,2,4,8,12}; 
    int numThreads = 5;
    int runs = 5;
    #ifdef _OPENMP
    cout << "OpenMP ONLINE" << endl;
    #else
    cout << "OpenMP OFFLINE" << endl;
    #endif

    system("mkdir results_openmp 2>nul");

    ofstream resultsFile("results_openmp/results.csv");
    resultsFile << "Size,Threads,Time_seconds\n";

    cout << "========================================" << endl;
    cout << "PARALLEL MATRIX MULTIPLYING (OPENMP)" << endl;
    cout << "========================================" << endl;
    cout << "Available threads: " << omp_get_max_threads() << endl;

    
    

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
        for(int j = 0; j < numThreads;j++){
         cout << "Multiply matrixes..." << endl;
        double total=0;
            for(int k=0; k < runs;k++){
       

        
        auto start = high_resolution_clock::now();
        auto C = multiply(A, B, threads[j]);
        auto end = high_resolution_clock::now();
        double seconds = duration_cast<milliseconds>(end - start).count() / 1000.0;
        total+=seconds;
        if (k==runs-1){
            saveMatrix(folder + "/C_"+intToString(threads[j])+".txt", C);

            
            cout << "Saved in folder: " << folder << endl;
        }
        

        

        
            }
        
        total=total/runs;

        resultsFile << n << "," << intToString(threads[j])<<","<< total << endl;
        cout <<"Threads: " << intToString(threads[j]) << "Time: " << total << "sec" << endl;
        
        }
    }

    resultsFile.close();

    cout << "\n========================================" << endl;
    cout << "Results of multiply: results.csv" << endl;
    cout << "Verificate results with: python verify.py" << endl;
    cout << "========================================" << endl;

    system("pause");
    return 0;
}