// hw2-2
#include <iostream>
#include <math.h>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <float.h>

#define ld double

using namespace std;

ld factorial(int n){
    if(n == 0) return 1;
    else return n*factorial(n-1);
}

ld c(int m, int n){
    // Assume always m > n
    // c(m, n) = m!/((m-n)!*n!)
    ld t1 = factorial(m);
    ld t2 = factorial(m-n);
    ld t3 = factorial(n);
    return t1/(t2*t3);
}

ld gammaFcn(int x){
    if(x <= 2) return 1;
    return ld((x-1)*gammaFcn(x-1));
}

ld betaFcn(ld p, int a, int b){
    return pow(p, a-1)*pow(1-p, b-1)*gammaFcn(a+b)/(gammaFcn(a)*gammaFcn(b));
}

ld binomial(ld p, int N, int m){
    ld t = c(N, m);
    return t*pow(p, m)*pow(1-p, N-m);
}

int main(int argc, char **argv){
    int a, b;
    a = stoi(argv[2]);
    b = stoi(argv[3]);
    int times = 0;

    ifstream ifs(string(argv[1]), ios::in);
        if(!ifs.is_open()){
        cout << "Error to open file" << endl;
    }else{
        string s;
        while(getline(ifs, s)){
            cout << "case " << ++times << ": "<< s << endl;
            int N = s.size(), m=0;
            ld p = 0;
            for(int i = 0; i < N; i++){
                m += (s[i]-'0');
            }
            p = ld(m)/ld(N);
            double likelihood = binomial(p, N, m);
            cout << "Likelihood: ";
            cout << setprecision(10) << likelihood << endl;
            cout << "Beta prior:    a = " << a << ", b = " << b << endl;
            a = a+m;
            b = b+N-m; 
            cout << "Beta posterior:    a = " << a << ", b = " << b << endl;
            cout << "\n";
        }
        ifs.close();
    }
    ifs.close();

    return 0;
}