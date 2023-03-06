#include <iostream>
#include <vector>
#include <stdlib.h>
#include <random>
#include <iomanip>
#include <cmath>
#include <ctime>
#include <fstream>
#include <sstream>
#include <string.h>

#define dd double
#define mat vector<vector<dd> >
#define pi 3.141592653589793

using namespace std;

typedef pair<dd, dd> point;

default_random_engine generator(time(NULL));
uniform_real_distribution<dd> distributionA(0.0, 1.0);
uniform_real_distribution<dd> distributionB(-1.0, 1.0);

mat zeroMatrix(int m, int n){   
    // get m by n zero matrix 
    mat z;
    int i, j;
    for(i = 0; i < m; i++){
        vector<dd> t;
        for(j = 0; j < n; j++){
            t.push_back(0);
        }
        z.push_back(t);
    }
    return z;
}

mat identityMatrix(int n){
    // get n by n identity matrix
    mat i;
    int j, k;
    for(j = 0; j < n; j++){
        vector<dd> t;
        for(k = 0; k < n; k++){
            if(k == j) t.push_back(1);
            else t.push_back(0);
        }
        i.push_back(t);
    }
    return i;
}

double dnorm(dd x, dd mean, dd var){
    double t = 1/(sqrt(2*var*pi));
    double t1 = (-1)*(x-mean)*(x-mean)/(2*var);
    double e = exp(t1);
    return t*e;
}

mat matrixTimes(mat A, dd times, int m, int n){
    // B = times*A, times is a scale
    mat B = zeroMatrix(m, n);
    int i, j;
    for(i = 0; i < m; i++){
        for(j = 0; j < n; j++){
            B[i][j] = times * A[i][j];
        }
    }
    return B;
}
mat matrixMul(mat A, mat B, int m, int n, int p){
    // A: m by n, B: n by p, C: m by p
    // C = AB
    int i, j, k;
    mat C = zeroMatrix(m, p);
    for(i = 0; i < m; i++){
        for(j = 0; j < n; j++){
            for(k = 0; k < p; k++){
                C[i][k] += A[i][j]*B[j][k];
            }
        }
    }
    return C;
}
mat matrixTrans(mat A, int m, int n){
    // A: m by n, B: n by m
    // B = A^T
    mat B = zeroMatrix(n, m);
    int i, j;
    for(i = 0; i < m; i++){
        for(j = 0; j < n; j++){
            B[j][i] = A[i][j];
        }
    }
    return B;
}
mat matrixSub(mat A, mat B, int m, int n){
    // C = A - B, A,B: m by n
    mat C = zeroMatrix(m, n);
    int i, j;
    for(i = 0; i < m; i++){
        for(j = 0; j < n; j++){
            C[i][j] = A[i][j] - B[i][j];
        }
    }
    return C;
}
mat matrixAdd(mat A, mat B, int m, int n){
    // C = A + B, A,B: m by n
    mat C= zeroMatrix(m, n);
    int i, j;
    for(i = 0; i < m; i++){
        for(j = 0; j < n; j++){
            C[i][j] = A[i][j] + B[i][j];
        }
    }
    return C;
}

pair<mat, mat> LUdecomposition(mat A, int n){
    int i, j, k;
    mat L, U;
    /* 
        L = | 1 0 0 |
            | 0 1 0 |
            | 0 0 1 |
        U = A
    */       
    L = identityMatrix(n);
    U = zeroMatrix(n,n);
    for(i = 0; i < n; i++){
        for(j = 0; j < n; j++){
            U[i][j] = A[i][j];
        }
    }
    for(i = 0; i < n; i++){
        for(j = i+1; j < n; j++){
            dd r = (U[j][i]/U[i][i]);
            L[j][i] = r;
            for(k = i; k < n ; k++){
                U[j][k] = U[j][k] - r*U[i][k];
            }
        }
    }
    return make_pair(L, U);

}
mat inverseU(mat U, int m){
    // iU = I, tmp = U;
    int i, j, k;
    
    mat iU = identityMatrix(m), tmp = zeroMatrix(m, m);
    for(i = 0; i < m; i++){
        for(j = 0; j < m; j++){
            tmp[i][j] = U[i][j];
        }
    }
    for(i = m-1; i >= 0; i--){
        dd p = tmp[i][i];
        for(j = m-1; j > -1; j--){
            tmp[i][j] = tmp[i][j] / p;
            iU[i][j] = iU[i][j] / p;
        }
        for(j = i-1; j >= 0; j--){
            p = tmp[j][i];    
            for(k = j; k <= m-1; k++){
                iU[j][k] = iU[j][k] - p*iU[i][k];
                tmp[j][k] = tmp[j][k] - p*tmp[i][k];
            } 
        }
    }
    return iU;
}

mat inverseL(mat L, int m){
    int i, j, k;
    // iL = I, tmp = L;
    mat iL = identityMatrix(m), tmp = zeroMatrix(m, m);
    for(i = 0; i < m; i++){
        for(j = 0; j < m; j++){
            tmp[i][j] = L[i][j];
        }
    }
    for(i = 0; i < m; i++){
        dd p = tmp[i][i];
        for(j = 0; j < m; j++){
            tmp[i][j] = tmp[i][j] / p;
            iL[i][j] = iL[i][j] / p;
        }
        for(j = i+1; j < m; j++){
            p = tmp[j][i];    
            for(k = j; k > -1; k--){
                iL[j][k] = iL[j][k] - p*iL[i][k];
                tmp[j][k] = tmp[j][k] - p*tmp[i][k];
            } 
        }
    }
    return iL;
}

mat InverseMatrix(mat A, int m){
    // iA = A^-1
    mat iA;
    int i, j;
    mat iL, iU;
    pair<mat, mat> LU;
    LU = LUdecomposition(A, m); // A = LU
    iL = inverseL(LU.first, m); // iL = L^-1
    iU = inverseU(LU.second, m); // iU = U^-1
    iA = matrixMul(iU, iL, m, m, m); // iA = U^-1 * L^-1
    return iA;
}


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
point randPoint(int n, dd a, mat w){
    int i;
    point p;
    dd x = distributionB(generator); // get x from (-1.0, 1.0)
    dd e = randNorm(0, a);
    dd y = 0.0;
    for(i = 0; i < n; i++){
        y += w[i][0]*pow(x, i);
    }
    y += e;
    p = make_pair(x, y);
    return p;
}

void generateDataTXT(vector<point> points, string filename){
    ofstream ofile;
    ofile.open(filename);
    for(vector<point>::iterator it = points.begin(); it != points.end(); ++it){
        ofile << to_string(it->first) << " " << to_string(it->second) << "\n";
    }
    ofile.close();
}
void generateGTtxt(mat mean, mat var, int n, dd a, string filenameU, string filenameL){
    int j;
    dd i;
    ofstream ofileU(filenameU);
    ofstream ofileL(filenameL);
    for(i = -2.0; i <= 2.0; i+=0.01){
        mat phix = zeroMatrix(n, 1);
        for(j = 0; j < n; j++){
            phix[j][0] = pow(i, j);
        }
        mat y = zeroMatrix(1,1);
        y = matrixMul(matrixTrans(mean, n, 1), phix, 1, n, 1);
        mat t = matrixTimes(identityMatrix(1), a, 1, 1);
        mat t1 = matrixMul(matrixTrans(phix, n, 1), matrixMul(var, phix, n, n, 1), 1, n, 1);
        mat v = matrixAdd(t, t1, 1, 1);
        ofileU << to_string(i) << " " << to_string(y[0][0]+v[0][0]) << "\n";
        ofileL << to_string(i) << " " << to_string(y[0][0]-v[0][0]) << "\n";
    }
    ofileL.close();
    ofileU.close();
}

void plot(vector<point> points10, vector<point> points50, vector<point> pointsAll, mat w10, mat w50, mat wAll, mat wO, mat var10, mat var50, mat varAll, int n, dd a){
    int i;
    string f10 = "", f50 = "", fAll = "", fO = "";
    generateDataTXT(points10, "data10.txt");
    generateDataTXT(points50, "data50.txt");
    generateDataTXT(pointsAll, "dataAll.txt");
    for(i = 0; i < n; i++){
        if(i == n-1){
            f10 += ("+"+to_string(w10[i][0])+"*x**"+to_string(i));
        }
        else f10 += (to_string(w10[i][0])+"*x**"+to_string(i)+"+");
    }
    for(i = 0; i < n; i++){
        if(i == n-1){
            f50 += ("+"+to_string(w50[i][0])+"*x**"+to_string(i));
        }
        else f50 += (to_string(w50[i][0])+"*x**"+to_string(i)+"+");
    }
    for(i = 0; i < n; i++){
        if(i == n-1){
            fAll += ("+"+to_string(wAll[i][0])+"*x**"+to_string(i));
        }
        else fAll += (to_string(wAll[i][0])+"*x**"+to_string(i)+"+");
    }
    for(i = 0; i < n; i++){
        if(i == n-1){
            fO += ("+"+to_string(wO[i][0])+"*x**"+to_string(i));
        }
        else fO += (to_string(wO[i][0])+"*x**"+to_string(i)+"+");
    }

    generateGTtxt(wAll, varAll, n, a, "gtAllU.txt", "gtAllL.txt");
    generateGTtxt(w10, var10, n, a, "gt10U.txt", "gt10L.txt");
    generateGTtxt(w50, var50, n, a, "gt50U.txt", "gt50L.txt");
    ofstream ofile;
    ofile.open("hw3Plot.gp");
    ofile << "set term qt\n";
    ofile << "set grid\n";
    ofile << "set xtics -2, 1\n";
    ofile << "set yrange [-20:20]\n";
    ofile << "set multiplot layout 2,2\n";
    ofile << "set title \"Ground truth\"\n";
    ofile << "plot [x=-2:2] " << fO << " title \"\" lt rgb \"black\" lw 1.2, ";
    ofile << fO << "+" << to_string(a) << " title \"\" lt rgb \"red\" lw 1.2, ";
    ofile << fO << "-" << to_string(a) << " title \"\" lt rgb \"red\" lw 1.2\n";
    ofile << "set title \"Predict result\"\n";
    ofile << "plot [x=-2:2] " << "\"dataAll.txt\"" << " using 1:2 title \"\" pt 52 ps 0.5 lc rgb \"#2767AA\", " ;
    ofile << fAll << " title \"\" lt rgb \"black\" lw 1.2, ";
    ofile << "\"gtAllU.txt\" title \"\" with lines lt rgb \"red\" lw 1.2, ";
    ofile << "\"gtAllL.txt\" title \"\" with lines lt rgb \"red\" lw 1.2 \n";
    ofile << "set title \"After 10 incomes\"\n";
    ofile << "plot [x=-2:2] " << "\"data10.txt\"" << " using 1:2 title \"\" pt 52 ps 0.5 lc rgb \"#2767AA\", " ;
    ofile << f10 << " title \"\" lt rgb \"black\" lw 1.2, ";
    ofile << "\"gt10U.txt\" title \"\" with lines lt rgb \"red\" lw 1.2, ";
    ofile << "\"gt10L.txt\" title \"\" with lines lt rgb \"red\" lw 1.2 \n";
    ofile << "set title \"After 50 incomes\"\n";
    ofile << "plot [x=-2:2] " << "\"data50.txt\"" << " using 1:2 title \"\" pt 52 ps 0.5 lc rgb \"#2767AA\", " ;
    ofile << f50 << " title \"\" lt rgb \"black\" lw 1.2, ";
    ofile << "\"gt50U.txt\" title \"\" with lines lt rgb \"red\" lw 1.2, ";
    ofile << "\"gt50L.txt\" title \"\" with lines lt rgb \"red\" lw 1.2 \n";
    ofile.close();

    system("gnuplot -p \"hw3Plot.gp\"");
}

int main(){
    cout << setprecision(10);
    int i, j, k;
    int n;
    dd a;
    dd b;
    mat w; // w: n by 1
    cout << "input b: ";
    cin >> b;
    cout << "input n: ";
    cin >> n;
    cout << "input a: ";
    cin >> a;
    cout << "input w: ";
    for(i = 0; i < n; i++){
        vector<dd> r;
        dd t;
        cin >> t;
        r.push_back(t);
        w.push_back(r);
    }
    cout << "n = " << n << ", a = " << a << ", w = ";
    for(i = 0; i < n; i++){
        cout << w[i][0] << " ";
    }
    cout << "\n";

    vector<point> after10incomes; // for ploting 10 points
    vector<point> after50incomes; // for ploting 50 points
    vector<point> allPoints; // for ploting all points
    mat mean10, mean50;
    mat var10, var50, varAll;
    mat posteriorMean, posteriorVar, priorMean, priorVar;
    priorMean = zeroMatrix(n, 1);
    priorVar = matrixTimes(identityMatrix(n), b, n, n);
    mat bI = matrixTimes(identityMatrix(n), b, n, n);
    mat X, y;

    cout << fixed;
    for(i = 1; i <= 20000; i++){
        point p = randPoint(n, a, w);

        cout << "Add data point: (" << p.first << ", " << p.second << ")\n\n";

        if(i <= 10) after10incomes.push_back(p);
        if(i <= 50) after50incomes.push_back(p);
        allPoints.push_back(p);
        vector<dd> t;
        mat phiX = zeroMatrix(1, n);
        for(j = 0; j < n; j++){
            t.push_back(pow(p.first, j));
            phiX[0][j] = pow(p.first, j);
        }
        X.push_back(t);

        vector<dd> ty;
        ty.push_back(p.second);
        y.push_back(ty);

        mat XT = matrixTrans(X, i, n); // XT = X^T
        mat s1 = matrixMul(XT, X, n, i, n); // s1 = X^T * X
        mat s2 = matrixTimes(s1, a, n, n); // s2 = a * X^T * X
        mat s3 = matrixAdd(bI, s2, n, n); // s3 = sigma_0 + a * X^T * X
        posteriorVar = InverseMatrix(s3, n); // posterier sigma = s3^-1

        mat m2 = matrixMul(XT, y, n, i, 1); // m2 = X^T * y
        mat m3 = matrixTimes(m2, a, n, 1); // m3 = a * X^T * y
        posteriorMean = matrixMul(posteriorVar, m3, n, n, 1); // (posterier sigma) * a * X^T * y

        cout << "Posterior mean:\n";
        for(j = 0; j < n; j++){
            cout << posteriorMean[j][0] << "\n";
        }
        cout << "\nPosterior variance:\n";
        for(j = 0; j < n; j++){
            for(k = 0; k < n; k++){
                cout << posteriorVar[j][k] << ",\t";
            }
            cout << "\n";
        }
        mat predictMean, predictVar;
        predictMean = matrixMul(phiX, posteriorMean, 1, n, 1);

        mat aI = matrixTimes(identityMatrix(1), a, 1, 1);
        mat pv1 = matrixMul(phiX, matrixMul(posteriorVar, matrixTrans(phiX, 1, n), n, n, 1), 1, n, 1);
        predictVar = matrixAdd(aI, pv1, 1, 1);
        cout << "\nPredict distribution ~ N(" << predictMean[0][0] << ", " << predictVar[0][0] << ")\n\n";

        if(i == 10){
            mean10 = posteriorMean;
            var10 = posteriorVar;
        }
        if(i == 50){
            mean50 = posteriorMean;
            var50 = posteriorVar;
        }

        dd dist = 0;
        for(j = 0; j < n; j++){
            dist += pow((posteriorMean[j][0] - priorMean[j][0]), 2);
        }
        if(sqrt(dist) <= 0.00001 && i >= 2000){
            break;
        };

        priorMean = posteriorMean;
        priorVar = posteriorVar;
    }
    varAll = posteriorVar;
    cout << "\n";
    plot(after10incomes, after50incomes, allPoints, mean10, mean50, posteriorMean, w, var10, var50, varAll, n, a);
    return 0;
}