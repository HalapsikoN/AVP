#include <iostream>
#include <Windows.h>
#include <ctime>

#define INNER_LINES_AMOUNT_1 4
#define INNER_COMMON_SIDE_AMOUNT 4
#define INNER_COLUMNS_AMOUNT_1 INNER_COMMON_SIDE_AMOUNT
#define INNER_LINES_AMOUNT_2 INNER_COMMON_SIDE_AMOUNT
#define INNER_COLUMNS_AMOUNT_2 4

#define MAIN_LINES_AMOUNT_1 50
#define MAIN_COMMON_SIDE_AMOUNT 50
#define MAIN_COLUMNS_AMOUNT_1 MAIN_COMMON_SIDE_AMOUNT
#define MAIN_LINES_AMOUNT_2 MAIN_COMMON_SIDE_AMOUNT
#define MAIN_COLUMNS_AMOUNT_2 50


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
			float firstMatrixNum = matrix1[i][j];
//#pragma loop(hint_parallel(4))
			for (int g = 0; g < n; ++g) {
				result[i][g] += firstMatrixNum * matrix2[j][g];
			}
		}
	}

	return result;
}

float** assemblerMultiMatrix4X4(float** matrix1, float** matrix2, int m, int n, int k) {

	float** result = createInnerMatrix(m, n);

	/*for (int i = 0; i < m; ++i) {
		for (int j = 0; j < k; ++j) {
			for (int g = 0; g < n; ++g) {
				result[i][g] += matrix1[i][j] * matrix2[j][g];
			}
		}
	}*/

	//_asm {
	//	pusha
	//	xor esi, esi		//refister for matrix1
	//	xor edi, edi		//register for matrix2
	//	xor ebp, ebp		//register for resultMatrix

	//	pxor MM7, MM7

	//	movq MM0, matrix1[esi]
	//	add esi, 8
	//	movq MM1, matrix1[esi]
	//	add esi, 8

	//	movd MM3, matrix2[edi]
	//	pssl MM3, 4						//shift left
	//	movd MM3, matrix2[edi + 64]
	//	movd MM4, matrix2[edi + 128]
	//	pssl MM4, 4
	//	movd MM4, matrix2[edi + 192]

	//	//mmx ne prokatit net norm funczii dlya peremnojenia


	//};

	return result;
}

float** intrncsMatrix(float** matrix1, float** matrix2, int m, int n, int k) {
	__m128 a_line, b_line, r_line;

	float** result = createInnerMatrix(m, n);

	for (int i = 0; i < m; ++i) {
			r_line = _mm_setzero_ps();
			for (int j = 0; j < k; j++)
			{
				b_line = _mm_load_ps(matrix2[j]);
				a_line = _mm_set1_ps(matrix1[i][j]);
				r_line = _mm_add_ps(_mm_mul_ps(a_line, b_line), r_line);
			}
			_mm_store_ps(result[i], r_line);
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

	float**** matrix3 = (float****)calloc(MAIN_LINES_AMOUNT_1, sizeof(float***));
	for (int i = 0; i < MAIN_LINES_AMOUNT_1; ++i) {
		matrix3[i]= (float***)calloc(MAIN_COLUMNS_AMOUNT_1, sizeof(float**));
		for (int j = 0; j < MAIN_COLUMNS_AMOUNT_1; ++j) {
			matrix3[i][j] = createInnerMatrix(INNER_LINES_AMOUNT_1, INNER_COLUMNS_AMOUNT_1);
			initializeMatrixWithRandomValues(matrix3[i][j], INNER_LINES_AMOUNT_1, INNER_COLUMNS_AMOUNT_1);
		}
	}

	float**** matrix4 = (float****)calloc(MAIN_LINES_AMOUNT_2, sizeof(float***));
	for (int i = 0; i < MAIN_LINES_AMOUNT_2; ++i) {
		matrix4[i] = (float***)calloc(MAIN_COLUMNS_AMOUNT_2, sizeof(float**));
		for (int j = 0; j < MAIN_COLUMNS_AMOUNT_2; ++j) {
			matrix4[i][j] = createInnerMatrix(INNER_LINES_AMOUNT_2, INNER_COLUMNS_AMOUNT_2);
			initializeMatrixWithRandomValues(matrix4[i][j], INNER_LINES_AMOUNT_2, INNER_COLUMNS_AMOUNT_2);
		}
	}

	float** resultMatrix1;
	QueryPerformanceCounter(&start);
	for (int i = 0; i < MAIN_LINES_AMOUNT_1; ++i) {
		for (int j = 0; j < MAIN_COLUMNS_AMOUNT_1; ++j) {
			for (int k = 0; k < MAIN_LINES_AMOUNT_2; ++k) {
				for (int l = 0; l < MAIN_COLUMNS_AMOUNT_2; ++l) {
					resultMatrix1= multiplyInnerMatrix_VLob(matrix3[i][j], matrix4[k][l], INNER_LINES_AMOUNT_1, INNER_COLUMNS_AMOUNT_2, INNER_COMMON_SIDE_AMOUNT);
				}
			}
		}
	}
	QueryPerformanceCounter(&finish);
	delay = (finish.QuadPart - start.QuadPart) * 1000.0f / frequency.QuadPart;
	cout << "Time v lob cicle in ms: " << delay << endl;

	float** resultMatrix2;
	QueryPerformanceCounter(&start);
	for (int i = 0; i < MAIN_LINES_AMOUNT_1; ++i) {
		for (int j = 0; j < MAIN_COLUMNS_AMOUNT_1; ++j) {
			for (int k = 0; k < MAIN_LINES_AMOUNT_2; ++k) {
				for (int l = 0; l < MAIN_COLUMNS_AMOUNT_2; ++l) {
					resultMatrix2 = multiplyInnerMatrix_optimized_only_line_loops(matrix3[i][j], matrix4[k][l], INNER_LINES_AMOUNT_1, INNER_COLUMNS_AMOUNT_2, INNER_COMMON_SIDE_AMOUNT);
				}
			}
		}
	}
	QueryPerformanceCounter(&finish);
	delay = (finish.QuadPart - start.QuadPart) * 1000.0f / frequency.QuadPart;
	cout << "Time vectorize cicle in ms: " << delay << endl;

	float** resultMatrix3;
	QueryPerformanceCounter(&start);
	for (int i = 0; i < MAIN_LINES_AMOUNT_1; ++i) {
		for (int j = 0; j < MAIN_COLUMNS_AMOUNT_1; ++j) {
			for (int k = 0; k < MAIN_LINES_AMOUNT_2; ++k) {
				for (int l = 0; l < MAIN_COLUMNS_AMOUNT_2; ++l) {
					resultMatrix3 = intrncsMatrix(matrix3[i][j], matrix4[k][l], INNER_LINES_AMOUNT_1, INNER_COLUMNS_AMOUNT_2, INNER_COMMON_SIDE_AMOUNT);
				}
			}
		}
	}
	QueryPerformanceCounter(&finish);
	delay = (finish.QuadPart - start.QuadPart) * 1000.0f / frequency.QuadPart;
	cout << "Time with using intrinsics in ms: " << delay << endl;

	cout << endl;
	showInnerMatrix(resultMatrix1, INNER_LINES_AMOUNT_1, INNER_COLUMNS_AMOUNT_2);
	cout << endl;
	showInnerMatrix(resultMatrix2, INNER_LINES_AMOUNT_1, INNER_COLUMNS_AMOUNT_2);
	cout << endl;
	showInnerMatrix(resultMatrix3, INNER_LINES_AMOUNT_1, INNER_COLUMNS_AMOUNT_2);
	cout << endl;

	float** matrix1 = createInnerMatrix(INNER_LINES_AMOUNT_1, INNER_COLUMNS_AMOUNT_1);
	float** matrix2 = createInnerMatrix(INNER_LINES_AMOUNT_2, INNER_COLUMNS_AMOUNT_2);
	initializeMatrixWithRandomValues(matrix1, INNER_LINES_AMOUNT_1, INNER_COLUMNS_AMOUNT_1);
	initializeMatrixWithRandomValues(matrix2, INNER_LINES_AMOUNT_2, INNER_COLUMNS_AMOUNT_2);

	QueryPerformanceCounter(&start);
	float** matrix5 = multiplyInnerMatrix_VLob(matrix1, matrix2, INNER_LINES_AMOUNT_1, INNER_COLUMNS_AMOUNT_2, INNER_COMMON_SIDE_AMOUNT);
	QueryPerformanceCounter(&finish);
	delay = (finish.QuadPart - start.QuadPart) * 1000.0f / frequency.QuadPart;
	cout << "Time \"v lob\" in ms: " << delay << endl;
	showInnerMatrix(matrix5, INNER_LINES_AMOUNT_1, INNER_COLUMNS_AMOUNT_2);


	float** matrix8 = createInnerMatrix(1000, 1000);
	float** matrix9 = createInnerMatrix(1000, 1000);
	initializeMatrixWithRandomValues(matrix8, 1000, 1000);
	initializeMatrixWithRandomValues(matrix9, 1000, 1000);
	QueryPerformanceCounter(&start);
	float** matrix6 = multiplyInnerMatrix_optimized_only_line_loops(matrix8, matrix9, 1000, 1000, 1000);
	QueryPerformanceCounter(&finish);
	delay = (finish.QuadPart - start.QuadPart) * 1000.0f / frequency.QuadPart;
	cout << "Time with line loops optimization in ms: " << delay << endl;
	//showInnerMatrix(matrix6, INNER_LINES_AMOUNT_1, INNER_COLUMNS_AMOUNT_2);
	QueryPerformanceCounter(&start);
	float** matrix10 = multiplyInnerMatrix_VLob(matrix8, matrix9, 1000, 1000, 1000);
	QueryPerformanceCounter(&finish);
	delay = (finish.QuadPart - start.QuadPart) * 1000.0f / frequency.QuadPart;
	cout << "Time \"v lob\" in ms: " << delay << endl;
	//showInnerMatrix(matrix5, INNER_LINES_AMOUNT_1, INNER_COLUMNS_AMOUNT_2);


	QueryPerformanceCounter(&start);
	float** matrix7 = intrncsMatrix(matrix1, matrix2, INNER_LINES_AMOUNT_1, INNER_COLUMNS_AMOUNT_2, INNER_COMMON_SIDE_AMOUNT);
	QueryPerformanceCounter(&finish);
	delay = (finish.QuadPart - start.QuadPart) * 1000.0f / frequency.QuadPart;
	cout << "Time with using intrinsics in ms: " << delay << endl;
	showInnerMatrix(matrix7, INNER_LINES_AMOUNT_1, INNER_COLUMNS_AMOUNT_2);

	/*float C[10000] = { 123.456 }, B[10000] = {678.678}, A[10000];

	QueryPerformanceCounter(&start);
	for (int i = 0; i < 10000; ++i)
		A[i] = 5 + C[i];
	QueryPerformanceCounter(&finish);
	delay = (finish.QuadPart - start.QuadPart) * 1000.0f / frequency.QuadPart;
	cout << "Time with line loops optimization in ms: " << delay << endl;

	QueryPerformanceCounter(&start);
#pragma loop(no_vector)
	for (int i = 0; i < 10000; ++i)
		A[i] = 5 + C[i];
	QueryPerformanceCounter(&finish);
	delay = (finish.QuadPart - start.QuadPart) * 1000.0f / frequency.QuadPart;
	cout << "Time with line loops optimization in ms: " << delay << endl;*/
	return 0;
}