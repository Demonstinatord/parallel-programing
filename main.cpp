#include <iostream>
#include <mpi.h>
#include <fstream>
#include <vector>
#include <random>
#include <string>
#include <sstream>
#include <iomanip>
#include <locale.h>
#include <direct.h>

using namespace std;

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

vector<double> flattenMatrix(const vector<vector<double>>& matrix) {
    int n = matrix.size();
    vector<double> flat(n * n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            flat[i * n + j] = matrix[i][j];
    return flat;
}

vector<vector<double>> unflattenMatrix(const vector<double>& flat, int n) {
    vector<vector<double>> matrix(n, vector<double>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            matrix[i][j] = flat[i * n + j];
    return matrix;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int sizes[] = { 200, 400, 800, 1200, 1600, 2000 };
    int numSizes = 6;

    if (world_rank == 0) {
        _mkdir("results_mpi");
        ifstream check("results_mpi/results.csv");
        bool file_exists = check.good();
        check.close();

        ofstream resultsFile("results_mpi/results.csv", ios::app);
        if (!file_exists) {
            resultsFile << "Size,Processes,Time_seconds\n";
        }
        resultsFile.close();

        cout << "Matrices parallel multiplying (MPI)" << endl;
        cout << "Processes ran: " << world_size << endl;
    }

    for (int i = 0; i < numSizes; ++i) {
        int n = sizes[i];

        // Расчет порций данных для каждого процесса (на случай, если n не делится на world_size)
        vector<int> sendcounts(world_size);
        vector<int> displs(world_size);
        int sum = 0;
        for (int p = 0; p < world_size; ++p) {
            int rows = n / world_size;
            if (p < n % world_size) rows++; // Остаток строк отдаем первым процессам
            sendcounts[p] = rows * n;
            displs[p] = sum;
            sum += sendcounts[p];
        }

        int local_size = sendcounts[world_rank];
        int rows_per_proc = local_size / n;

        vector<double> flat_A, flat_B(n * n), flat_C;
        vector<double> local_A(local_size);
        vector<double> local_C(local_size);
        double elapsed = 0.0;

        if (world_rank == 0) {
            string folder = "results_mpi/data_" + intToString(n);
            _mkdir(folder.c_str());
            cout << " Size " << n << "x" << n << endl;
            cout << "Generate matrices..." << endl;

            auto A = generateMatrix(n);
            auto B = generateMatrix(n);
            saveMatrix(folder + "/A.txt", A);
            saveMatrix(folder + "/B.txt", B);

            flat_A = flattenMatrix(A);
            flat_B = flattenMatrix(B);
            flat_C.resize(n * n);
            cout << "Multiply... (processes: " << world_size << ")" << endl;
        }

        // 1. Рассылаем матрицу B всем процессам
        MPI_Bcast(flat_B.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // 2. Рассылаем уникальные строки матрицы A процессам
        MPI_Scatterv(flat_A.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                     local_A.data(), local_size, MPI_DOUBLE, 
                     0, MPI_COMM_WORLD);

        // Синхронизация перед замером времени
        MPI_Barrier(MPI_COMM_WORLD);
        double start_time = MPI_Wtime();

        // 3. Вычисление локальной части матрицы C (Оптимизировано по кэшу: i -> k -> j)
        for (int i_row = 0; i_row < rows_per_proc; ++i_row) {
            // Инициализируем нулями строку локального результата
            for (int j = 0; j < n; ++j) {
                local_C[i_row * n + j] = 0.0;
            }
            // Выполняем кэш-оптимизированное умножение
            for (int k = 0; k < n; ++k) {
                double a_val = local_A[i_row * n + k];
                for (int j = 0; j < n; ++j) {
                    local_C[i_row * n + j] += a_val * flat_B[k * n + j];
                }
            }
        }

        double end_time = MPI_Wtime();
        elapsed = end_time - start_time;

        // Находим максимальное время выполнения среди всех процессов
        double max_elapsed = 0.0;
        MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        // 4. Собираем посчитанные строки обратно в главную матрицу flat_C
        MPI_Gatherv(local_C.data(), local_size, MPI_DOUBLE,
                    flat_C.data(), sendcounts.data(), displs.data(), MPI_DOUBLE, 
                    0, MPI_COMM_WORLD);

        if (world_rank == 0) {
            auto C = unflattenMatrix(flat_C, n);
            string folder = "results_mpi/data_" + intToString(n);
            saveMatrix(folder + "/C.txt", C);
            
            cout << "Time MPI: " << max_elapsed << " s" << endl;
            
            ofstream resultsFile("results_mpi/results.csv", ios::app);
            resultsFile << n << "," << world_size << "," << max_elapsed << "\n";
            resultsFile.close();
        }
    }

    if (world_rank == 0) {
        cout << "Results saved in results_mpi/results.csv" << endl;
    }

    MPI_Finalize();
    return 0;
}
