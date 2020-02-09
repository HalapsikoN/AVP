#include <iostream>
#include <Windows.h>

//make INNER_COLUMNS_AMOUNT_1 and INNER_LINES_AMOUNT_2 the same number
#define INNER_LINES_AMOUNT_1 1000
#define INNER_COLUMNS_AMOUNT_1 500
#define INNER_LINES_AMOUNT_2 500
#define INNER_COLUMNS_AMOUNT_2 1000

#define MAIN_LINES_AMOUNT_1 59
#define MAIN_COLUMNS_AMOUNT_1 59
#define MAIN_LINES_AMOUNT_2 59
#define MAIN_COLUMNS_AMOUNT_2 59


using namespace std;

float** createMatrix(int a, int b) {
	float** matrix = (float**)calloc(a, sizeof(float*));
	for (int i = 0; i < a; ++i) {
		matrix[i] = (float*)calloc(b, sizeof(float));
	}
	return matrix;
}

void fillMatrixWithNumber(float** matrix, int a, int b, float number) {
	for (int i = 0; i < a; ++i) {
		for (int j = 0; j < b; ++j) {
			matrix[i][j] = number;
		}
	}
}

float** multiplyMatrixVLob(float** matrix1, float** matrix2, int m, int n, int k) {

	float** result = createMatrix(m, k);

	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < k; ++j) {
			//float temp = 0;
			for (int g = 0; g < n; ++g) {
				result[i][j] += matrix1[i][g] * matrix2[g][j];
			}
		}
	}

	return result;
}

void showMatrix(float** matrix, int a, int b) {
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

	float** matrix1=createMatrix(INNER_LINES_AMOUNT_1, INNER_COLUMNS_AMOUNT_1);
	
	fillMatrixWithNumber(matrix1, INNER_LINES_AMOUNT_1, INNER_COLUMNS_AMOUNT_1, 2);

	float** matrix2 = createMatrix(INNER_LINES_AMOUNT_2, INNER_COLUMNS_AMOUNT_2);

	fillMatrixWithNumber(matrix2, INNER_LINES_AMOUNT_2, INNER_COLUMNS_AMOUNT_2, 3);

	//cout<<sizeof(float)<<endl;
	QueryPerformanceCounter(&start);
	float** matrix3 = multiplyMatrixVLob(matrix1, matrix2, INNER_LINES_AMOUNT_1, INNER_COLUMNS_AMOUNT_1, INNER_COLUMNS_AMOUNT_2);
	QueryPerformanceCounter(&finish);

	//showMatrix(matrix3, INNER_LINES_AMOUNT_1, INNER_COLUMNS_AMOUNT_2);

	delay = (finish.QuadPart - start.QuadPart) * 1000.0f / frequency.QuadPart;
	cout << "Time \"v lob\" in ms: " << delay << endl;

	return 0;
}