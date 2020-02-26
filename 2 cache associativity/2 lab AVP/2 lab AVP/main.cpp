#include <iostream>
#include <Windows.h>
#include <ctime>

#define CASHL1 32768
#define CASHL2
#define CASHL3
#define ELEMENT_SIZE 8
#define LINE_SIZE 64
#define ELEMENTS_AMOUNT_IN_WAY 64
#define NUMBER_OF_WAYES 8
#define NUMBER_OF_SETS 64
#define NMAX 20
#define SIZE_ARRAY CASHL1 / sizeof(long long)*NMAX

using namespace std;

int main() {
	LARGE_INTEGER frequency, start, finish;
	float delay;
	QueryPerformanceFrequency(&frequency);
	srand(static_cast <unsigned> (time(0)));

	long long memory[SIZE_ARRAY];
	long long temp=0;

	for (int i = 0; i < SIZE_ARRAY; ++i) {
		memory[i] = 0;
	}
	
	for (int n = 1; n <= NMAX; ++n) {
		
		int counter = 0;

		for (int i = 0; i < ELEMENTS_AMOUNT_IN_WAY / n; ++i) {
			int j=0;
			for (j = 0; j < n-1; ++j) {
				memory[j*ELEMENTS_AMOUNT_IN_WAY*ELEMENT_SIZE + i*ELEMENT_SIZE] = (j+1)*ELEMENTS_AMOUNT_IN_WAY*ELEMENT_SIZE +i*ELEMENT_SIZE;
			}
			memory[j*ELEMENTS_AMOUNT_IN_WAY*ELEMENT_SIZE + i*ELEMENT_SIZE] = (i +1)*ELEMENT_SIZE;
		}

		temp = 0;

		QueryPerformanceCounter(&start);
		for (int i = 0; i < (SIZE_ARRAY+SIZE_ARRAY)*10; ++i) {
			temp = memory[temp];
			counter++;
		}
		QueryPerformanceCounter(&finish);
		delay = (finish.QuadPart - start.QuadPart) * 1000.0f / frequency.QuadPart;
		cout << "for n ("<<n<<") in ms: " <<delay << endl;
		//cout << counter << endl;

	}


	return 0;
}