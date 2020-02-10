#include <iostream>
#include <Windows.h>
#include <ctime>

#define INNER_LINES_AMOUNT_1 1000
#define INNER_COMMON_SIDE_AMOUNT 700
#define INNER_COLUMNS_AMOUNT_1 INNER_COMMON_SIDE_AMOUNT
#define INNER_LINES_AMOUNT_2 INNER_COMMON_SIDE_AMOUNT
#define INNER_COLUMNS_AMOUNT_2 2000

#define MAIN_LINES_AMOUNT_1 59
#define MAIN_COMMON_SIDE_AMOUNT 59
#define MAIN_COLUMNS_AMOUNT_1 MAIN_COMMON_SIDE_AMOUNT
#define MAIN_LINES_AMOUNT_2 MAIN_COMMON_SIDE_AMOUNT
#define MAIN_COLUMNS_AMOUNT_2 59


using namespace std;

float** createInnerMatrix(int a, int b) {
	float** matrix = (float**)calloc(a, sizeof(float*));
	for (int i = 0; i < a; ++i) {
		matrix[i] = (float*)calloc(b, sizeof(float));
	}
	return matrix;
}

void initializeMatrixWithRandomValues(float** matrix, int a, int b) {
	for (int i = 0; i < a; ++i) {
		for (int j = 0; j < b; ++j) {
			matrix[i][j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);;
		}
	}
}

float** multiplyInnerMatrix_VLob(float** matrix1, float** matrix2, int m, int n, int k) {

	float** result = createInnerMatrix(m, n);

	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			for (int g = 0; g < k; ++g) {
				result[i][j] += matrix1[i][g] * matrix2[g][j];
			}
		}
	}

	return result;
}

float** multiplyInnerMatrix_optimized_only_line_loops(float** matrix1, float** matrix2, int m, int n, int k) {

	float** result = createInnerMatrix(m, n);

	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < k; ++j) {
			for (int g = 0; g < n; ++g) {
				result[i][g] += matrix1[i][j] * matrix2[j][g];
			}
		}
	}

	return result;
}

void showInnerMatrix(float** matrix, int a, int b) {
	for (int i = 0; i < a; ++i) {
		for (int j = 0; j < b; ++j) {
			cout << matrix[i][j] << " ";
		}
		cout << endl;
	}
}

int main() {

	LARGE_INTEGER frequency, start, finish;
	float delay;
	QueryPerformanceFrequency(&frequency);
	srand(static_cast <unsigned> (time(0)));

	float** matrix1 = createInnerMatrix(INNER_LINES_AMOUNT_1, INNER_COLUMNS_AMOUNT_1);
	float** matrix2 = createInnerMatrix(INNER_LINES_AMOUNT_2, INNER_COLUMNS_AMOUNT_2);
	initializeMatrixWithRandomValues(matrix1, INNER_LINES_AMOUNT_1, INNER_COLUMNS_AMOUNT_1);
	initializeMatrixWithRandomValues(matrix2, INNER_LINES_AMOUNT_2, INNER_COLUMNS_AMOUNT_2);

	QueryPerformanceCounter(&start);
	float** matrix3 = multiplyInnerMatrix_VLob(matrix1, matrix2, INNER_LINES_AMOUNT_1, INNER_COLUMNS_AMOUNT_2, INNER_COMMON_SIDE_AMOUNT);
	QueryPerformanceCounter(&finish);
	delay = (finish.QuadPart - start.QuadPart) * 1000.0f / frequency.QuadPart;
	cout << "Time \"v lob\" in ms: " << delay << endl;

	QueryPerformanceCounter(&start);
	float** matrix4 = multiplyInnerMatrix_optimized_only_line_loops(matrix1, matrix2, INNER_LINES_AMOUNT_1, INNER_COLUMNS_AMOUNT_2, INNER_COMMON_SIDE_AMOUNT);
	QueryPerformanceCounter(&finish);
	delay = (finish.QuadPart - start.QuadPart) * 1000.0f / frequency.QuadPart;
	cout << "Time with line loops optimization in ms: " << delay << endl;

	return 0;
}