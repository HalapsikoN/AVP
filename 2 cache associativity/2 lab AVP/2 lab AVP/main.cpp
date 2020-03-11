#include <iostream>
#include <Windows.h>
#include <ctime>
#include <stdio.h>
#include <intrin.h>

//Intel Core I-7 7700HQ
//L1 - size:32 kB, 8-way, 64-byte line size
//L2 - size:256 kB, 4-way, 64-byte line size
//L3 - size:6 MB, 12-way, 64-byte line size

#define ELEMENT_SIZE 8
#define NMAX 20

//<----------------------------------L1------------------------------>
//#define CASH 32768
//#define NUMBER_OF_WAYES 8
//#define CASH_LINE_SIZE 64
//
//#define ELEMENTS_AMOUNT_IN_WAY CASH/CASH_LINE_SIZE/NUMBER_OF_WAYES
//#define OFFSET CASH/CASH_LINE_SIZE
//#define SIZE_ARRAY CASH/sizeof(long long) *NMAX
//<----------------------------------L2------------------------------>
#define CASH 262144
#define NUMBER_OF_WAYES 4
#define CASH_LINE_SIZE 64

#define ELEMENTS_AMOUNT_IN_WAY CASH/CASH_LINE_SIZE/NUMBER_OF_WAYES
#define OFFSET CASH/CASH_LINE_SIZE
#define SIZE_ARRAY CASH/sizeof(long long) *NMAX
//<----------------------------------L3------------------------------>
//#define CASH 6291456
//#define NUMBER_OF_WAYES 12
//#define CASH_LINE_SIZE 64
//
//#define ELEMENTS_AMOUNT_IN_WAY CASH/CASH_LINE_SIZE/NUMBER_OF_WAYES
//#define OFFSET CASH/CASH_LINE_SIZE
//#define SIZE_ARRAY CASH/sizeof(long long) *NMAX

using namespace std;

int main() {
	LARGE_INTEGER frequency, start, finish;
	float delay;
	QueryPerformanceFrequency(&frequency);
	srand(static_cast <unsigned> (time(0)));
	unsigned __int64 st, end;

	long long* memory=(long long*)calloc(SIZE_ARRAY, sizeof(long long));
	long long temp=0;
	
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
		
		for (int i = 0; i < (SIZE_ARRAY)*10; ++i) {
			temp = memory[temp];
			counter++;
		}
		
		QueryPerformanceCounter(&finish);
		delay = (finish.QuadPart - start.QuadPart) * 1000.0f / frequency.QuadPart;
		cout << "for n ("<<n<<") in ms: " <<delay << endl;
		
	}
	memory[temp] = 0;
	
	system("pause");
	return 0;
}