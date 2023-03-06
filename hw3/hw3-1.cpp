#include <iostream>
#include <vector>
#include <stdlib.h>
#include <random>
#include <iomanip>
#include <cmath>
#include <ctime>
#include <fstream>
#include <sstream>

#define dd double
#define pi 3.141592653589793

using namespace std;

typedef pair<dd, dd> point;

default_random_engine generator(time(NULL));
uniform_real_distribution<dd> distributionA(0.0, 1.0);
uniform_real_distribution<dd> distributionB(-1.0, 1.0);


dd randNorm(dd mean, dd variance){
    dd U=0;
    int i;
    for(i =0; i < 12; i++){
        dd u = distributionA(generator);
        U += u;
    }
    U -= 6;
    return sqrt(variance)*U+mean;
}

point randPoint(int n, dd a, vector<dd> w){
    int i;
    point p;
    dd x = distributionB(generator);
    dd e = randNorm(0, a);
    dd y = 0.0;
    for(i = 0; i < n; i++){
        y += w[i]*pow(x, i);
    }
    y += e;
    p = make_pair(x, y);
    return p;
}

int main(int argc, char **argv){
    cout << setprecision(15);
    if(argv[1][0] - 'a' == 0){
        dd mean = 3.0;
        dd var = 5.0;
        cout << "input mean: ";
        cin >> mean;
        cout << "input variance: ";
        cin >> var;
        cout << "Data point source function: N(" << mean << ", " << var << ")\n\n";
        dd x = randNorm(mean, var);
        cout << "Data point : " << x << "\n";
    }else{
        int i;
        int n;
        dd a;
        vector<dd> w;
        cout << "input n: ";
        cin >> n;
        cout << "input a: ";
        cin >> a;
        cout << "input w: ";
        for(i = 0; i < n; i++){
            dd t;
            cin >> t;
            w.push_back(t);
        }
        cout << "n = " << n << ", a = " << a << ", w = ";
        for(i = 0; i < n; i++){
            cout << w[i] << " ";
        }
        cout << "\n";
        point p = randPoint(n, a, w);
        cout << "(x, y) = (" << p.first << ", " << p.second << ")\n";
    }

    return 0;
}