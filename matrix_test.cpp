#include <vector>
#include <random>
#include <chrono>
#include <iostream>

#define N 4096
#define d 4096
#define T 64

void matMulNaiveTranspose(std::vector<float> const &Q, std::vector<float> const &K,
                          std::vector<float> &QK_t)
{
    for (int row = 0; row < N; row++)
    {
        for (int inner = 0; inner < d; inner++)
        {
            float qVal = Q[row * d + inner];
            for (int col = 0; col < N; col++)
            {
                QK_t[row * N + col] += qVal * K[inner * N + col];
            }
        }
    }
}

void matMulTiling(std::vector<float> const &Q, std::vector<float> const &K,
                  std::vector<float> &QK_t)
{
    for (int trow = 0; trow < N; trow += T)
    {
        for (int tcol = 0; tcol < N; tcol += T)
        {
            for (int tinner = 0; tinner < d; tinner += T)
            {
                for (int row = trow; row < std::min(trow + T, N); row++)
                {
                    for (int col = tcol; col < std::min(tcol + T, N); col++)
                    {
                        float sum = QK_t[row * N + col];
                        for (int inner = tinner; inner < std::min(tinner + T, d); inner++)
                        {
                            sum += Q[row * d + inner] * K[col * d + inner];
                        }
                        QK_t[row * N + col] = sum;
                    }
                }
            }
        }
    }
}

void matMulNaive(std::vector<float> const &Q, std::vector<float> const &K,
                 std::vector<float> &QK_t)
{
    for (int row = 0; row < N; row++)
    {
        for (int col = 0; col < N; col++)
        {
            float val = 0.0;
            for (int inner = 0; inner < d; inner++)
            {
                val += Q[row * d + inner] * K[col * d + inner];
            }
            QK_t[row * N + col] = val;
        }
    }
}

bool areEqual(const std::vector<float> &vec1, const std::vector<float> &vec2, float tolerance = 1e-5)
{
    if (vec1.size() != vec2.size())
    {
        return false;
    }

    for (size_t i = 0; i < vec1.size(); ++i)
    {
        if (std::fabs(vec1[i] - vec2[i]) > tolerance)
        {
            std::cout << "QK_t_n[i]: " << std::to_string(vec1[i]) << std::endl;
            std::cout << "QK_t_t[i]: " << std::to_string(vec2[i]) << std::endl;
            return false;
        }
    }

    return true;
}

int main()
{
    // Initialize a random number generator
    std::random_device rd;                               // Obtain a random number from hardware
    std::mt19937 gen(rd());                              // Seed the generator
    std::uniform_real_distribution<float> dis(0.0, 1.0); // Define the range for random floats

    // Create the vector and populate it with random floats
    std::vector<float> Q(N * d);
    std::vector<float> K(N * d);
    std::vector<float> QK_t_n(N * N);  // naive
    std::vector<float> QK_t_nt(N * N); // naive transpose
    std::vector<float> QK_t_t(N * N);  // tiling

    for (int i = 0; i < N * d; ++i)
    {
        Q[i] = dis(gen); // Assign random float values
        K[i] = dis(gen); // Assign random float values
    }

    using std::chrono::duration_cast;
    using std::chrono::high_resolution_clock;
    using std::chrono::milliseconds;

    auto t1 = high_resolution_clock::now();
    matMulNaive(Q, K, QK_t_n);
    auto t2 = high_resolution_clock::now();
    auto ms_int = duration_cast<milliseconds>(t2 - t1);
    std::cout << "----NAIVE MAT MUL----\n";
    std::cout << ms_int.count() << "ms\n\n";

    t1 = high_resolution_clock::now();
    matMulNaiveTranspose(Q, K, QK_t_nt);
    t2 = high_resolution_clock::now();
    ms_int = duration_cast<milliseconds>(t2 - t1);
    std::cout << "----TRANSPOSE MAT MUL----\n";
    std::cout << ms_int.count() << "ms\n\n";

    t1 = high_resolution_clock::now();
    matMulTiling(Q, K, QK_t_t);
    t2 = high_resolution_clock::now();
    ms_int = duration_cast<milliseconds>(t2 - t1);
    std::cout << "----TILING MAT MUL----\n";
    std::cout << ms_int.count() << "ms\n\n";

    if (areEqual(QK_t_n, QK_t_t))
    {
        std::cout << "Naive & tiling vectors are equal." << std::endl;
    }
    else
    {
        std::cout << "Naive & tiling vectors are not equal." << std::endl;
    }

    return 0;
}
