#include <iostream>
#include <Windows.h>
#include <ctime>

#include <stdio.h>
#include <intrin.h>

//#pragma intrinsic(__rdtsc)


#define CASHL1 32768
#define CASHL2
#define CASHL3
#define ELEMENT_SIZE 8
//#define LINE_SIZE 64
#define ELEMENTS_AMOUNT_IN_WAY 64
//#define NUMBER_OF_WAYES 8
//#define NUMBER_OF_SETS 64
#define OFFSET 512
#define NMAX 20
#define SIZE_ARRAY CASHL1/sizeof(long long) *NMAX


using namespace std;

int main() {
	LARGE_INTEGER frequency, start, finish;
	float delay;
	QueryPerformanceFrequency(&frequency);
	srand(static_cast <unsigned> (time(0)));
	unsigned __int64 st, end;

	long long memory[SIZE_ARRAY];
	long long temp=0;

	for (int i = 0; i < SIZE_ARRAY; ++i) {
		memory[i] = 0;
	}
	
	for (int n = 1; n <= NMAX; ++n) {
		
		int counter = 0;

		int tempOffset = OFFSET / n;

		for (int i = 0; i < tempOffset; ++i) {
			int j=0;
			for (j = 0; j < n-1; ++j) {
				memory[j*ELEMENTS_AMOUNT_IN_WAY*ELEMENT_SIZE + i*ELEMENT_SIZE] = (j+1)*ELEMENTS_AMOUNT_IN_WAY*ELEMENT_SIZE +i*ELEMENT_SIZE;
			}
			memory[j*ELEMENTS_AMOUNT_IN_WAY*ELEMENT_SIZE + i*ELEMENT_SIZE] = (i +1)*ELEMENT_SIZE;
		}

		temp = 0;

		QueryPerformanceCounter(&start);
		//st = __rdtsc();
		for (int i = 0; i < (SIZE_ARRAY)*100; ++i) {
			temp = memory[temp];
			counter++;
		}
		//cout << counter << endl;
		//end = __rdtsc();
		QueryPerformanceCounter(&finish);
		delay = (finish.QuadPart - start.QuadPart) * 1000.0f / frequency.QuadPart;
		cout << "for n ("<<n<<") in ms: " <<delay << endl;
		//cout << "for n (" << n << ") in ms: " << end-st << endl;
		//cout << counter << endl;

	}
	
	
	
	
	
	system("pause");
	return 0;
}