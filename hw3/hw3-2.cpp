#include <iostream>
#include <vector>
#include <stdlib.h>
#include <random>
#include <iomanip>
#include <cmath>
#include <ctime>

#define dd double

using namespace std;

default_random_engine generator(time(NULL));
uniform_real_distribution<dd> distribution(0.0, 1.0);

dd randNorm(dd mean, dd variance){
    dd U=0;
    int i;
    for(i =0; i < 12; i++){
        dd u = distribution(generator);
        U += u;
    }
    U -= 6;
    return sqrt(variance)*U+mean;
}

int main(){
    cout << fixed << setprecision(15); 
    dd mean = 3.0;
    dd var = 5.0;
    cout << "input mean: ";
    cin >> mean;
    cout << "input variance: ";
    cin >> var;
    cout << "Data point source function: N(" << mean << ", " << var << ")\n\n";
    dd x = randNorm(mean, var);
    dd meanHead=x;
    dd varHead = 0.0;
    dd K = 3.0;
    dd EX = x-K, EX2 = (x-K)*(x-K);
    cout << "Add data point: " << x << "\n";
    cout << "Mean = " << meanHead << "\tVariance = " << varHead << "\n\n"; 
    int i;
    dd dm = 99, dv = 99;
    for(i = 2; i < 500000; i++){
        dd prevMean = meanHead;
        x = randNorm(mean, var);
        dm = ((i-1)*meanHead+x)/i - meanHead;
        meanHead = ((i-1)*meanHead+x)/i;
        EX += (x-K);
        EX2 += (x-K)*(x-K);
        dv = (EX2-(EX*EX)/i)/(i-1) - varHead;
        varHead = (EX2-(EX*EX)/i)/(i-1);
        cout << "Add data point: " << x << "\n";
        cout << "Mean = " << meanHead << "\tVariance = " << varHead << "\n\n";
        if(abs(meanHead - prevMean) <= 0.0001 && i > 1000) break;
    }
    return 0;
}