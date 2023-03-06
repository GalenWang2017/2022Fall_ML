// hw 4-1
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

// initial parameter

/*
    x: n by 3
           |1 x1 y1|
    and x= |1 x2 y2|
           |...    |
           |1 xn yn|

    initial w
    w = [0 0 0]^T

    w^T * x_i = w0 + w1 * xi + w2 * yi, x_i is row i of x
*/


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

dd det(mat A, int m){
    if(m == 1) return A[0][0];
    if(m == 2){
        return (A[0][0]*A[1][1]-A[0][1]*A[1][0]);
    }else{
        dd d = 0;
        for(int c = 0; c < m; c++){
            mat B = zeroMatrix(m, m);
            int t1 = 0;
            for(int i = 1; i < m; i++){
                int t2 = 0;
                for(int j = 0; j < m; j++){
                    if(j != c){
                        B[t1][t2] = A[i][j];
                        t2++;
                    }
                }
                t1++;
            }
            d += pow(-1, c) * A[0][c] *det(B, m-1);
        }
        return d;
    }
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

dd rnorm(dd mean, dd variance){
    normal_distribution<dd> dt(mean, sqrt(variance));
    dd rlt = dt(generator);
    return rlt;
}

dd norm2(mat A, int m, int n){
    // return ||A||, A: m by n
    int i, j;
    dd rlt = 0;
    for(i = 0; i < m; i++){
        for(j = 0; j < n; j++){
            dd t = A[i][j] * A[i][j];
            rlt += t;
        }
    }
    return sqrt(rlt);
}

pair<vector<point>, vector<point> > splitData(vector<point> p, vector<int> labels){
    int i;
    vector<point> p1, p2;
    for(i = 0; i < p.size(); i++){
        if(labels[i] == 0) p1.push_back(p[i]);
        else p2.push_back(p[i]);
    }
    return make_pair(p1, p2);
}

mat f(mat wTx, int n){
    // wTx = n by 1
    /*                                          1
        return rlt: n by 1, rlt[i][0] =  ---------------
                                          1+exp(-wTx[i])
    */
    int i;
    mat rlt = zeroMatrix(n, 1);
    for(int i = 0; i < n; i++){
        dd temp = 1+exp(0-wTx[i][0]);
        rlt[i][0] = 1/temp;
    }
    return rlt;
}

mat D(mat wtx, int n){
    // wtx: n by 1;
    /*                                               exp(-wtx[i])
        return rlt: diagonal matrix, rlt[i][i] = --------------------
                                                 (1 + exp(-wtx[i]))^2
    */
    mat rlt = identityMatrix(n);
    int i;
    for(i = 0; i < n; i++){
        dd t = exp(0-wtx[i][0]);
        rlt[i][i] = t/((1+t)*(1+t)); 
    }
    return rlt;
}

mat GradientDescent(mat X, mat w_zero, mat Y, dd lambda, int n){
    int i, j;
    // X: n by 3, w_old: 3 by 1
    mat w_old = w_zero;
    mat w_new = zeroMatrix(3, 1);
    int c = 0;
    dd prevNormOfGradient = 0;
    mat xt = matrixTrans(X, n, 3); // xt = X^T
    while(1){
        mat wtx = matrixMul(X, w_old, n, 3, 1); // wtx: n by 1
        mat fwtx = f(wtx, n); // f(wtx)
        mat t1 = matrixSub(Y, fwtx, n, 1); // t1 = f(wtx)-Y
        mat gradient = matrixMul(xt, t1, 3, n, 1);
        w_new = matrixAdd(w_old, matrixTimes(gradient, lambda, 3, 1), 3, 1);
        dd normOfGradient = norm2(gradient, 3, 1);
        if((prevNormOfGradient - normOfGradient) <= 15 && lambda != 0){
            if(lambda-0.001 > 0.2) lambda -= 0.001;
        }
        prevNormOfGradient = normOfGradient;
        w_old = w_new;

        if(normOfGradient <= 0.1 && c >= 10000) break;
        c += 1;
        if(c > 30000) break;
    }
    return w_new;
}

mat NewtonMethod(mat X, mat w_zero, mat Y, int n){
    int i, j;
    mat w_old = w_zero;
    mat w_new = zeroMatrix(3, 1);
    mat xt = matrixTrans(X, n, 3);
    int c = 0;
    while(1){
        mat wtx = matrixMul(X, w_old, n, 3, 1); // wtx: n by 1
        mat dmatrix = D(wtx, n); // dmatrix: n by n
        mat fwtx = f(wtx, n); // fwtx: n by 1
        mat phiTDphi = matrixMul(xt, matrixMul(dmatrix, X, n, n, 3), 3, n, 3); // X^T * dmatrix * X
        mat t1 = matrixSub(fwtx, Y, n, 1); // f(w^T * X)-Y
        mat gradient = matrixMul(xt, t1, 3, n, 1); // X^T * (f(w^T * X)-Y)
        dd normOfGradient = norm2(gradient, 3, 1);
        if(det(phiTDphi, 3) != 0){
            // if phiTDphi is nonsingular
            mat invOfphiTDphi = InverseMatrix(phiTDphi, 3);
            mat t2 = matrixMul(invOfphiTDphi, gradient, 3, 3, 1);
            w_new = matrixSub(w_old, t2, 3, 1);
            normOfGradient = norm2(t2, 3, 1);
        }else{ 
            // if phiTDphi is singular
            w_new = matrixSub(w_old, gradient, 3, 1);
        }
        if(normOfGradient <= 0.1 && c >= 10000) break;
        c += 1;
        w_old = w_new;
        if(c > 30000) break;
    }
    return w_new;
}

vector<int> predict(mat X, mat w, int n){
    int i, j;
    vector<int> predictRlt;
    // X: n by 3
    mat wtx = matrixMul(X, w, n, 3, 1); // wtx: n by 1
    mat fwtx = f(wtx, n);
    for(i = 0; i < n; i++){
        if(fwtx[i][0] > 0.5) predictRlt.push_back(1);
        else predictRlt.push_back(0);
    }
    return predictRlt;
}

void confusionMatrix(vector<int> prediction, vector<int> labels, int n){
    dd tp = 0, tn = 0, fp = 0, fn = 0;
    for(int i = 0;i < n; i++){
        if(prediction[i] == 0){
            if(labels[i] == 0){
                tp += 1;
            }else{
                fp += 1;
            }
        }else{
            if(labels[i] == 1){
                tn += 1;
            }else{
                fn += 1;
            }            
        }
    }
    cout << setprecision(0);
    cout << "\t\t  Predict cluster 1 \t Predict cluster 2\n";
    cout << "Is cluster 1 \t\t" << tp << "\t\t\t" << fn << "\n"; 
    cout << "Is cluster 2 \t\t" << fp << "\t\t\t" << tn << "\n";
    cout << setprecision(6); 
    cout << "\nSensitivity (Successfully predict cluster 1): " << tp/(tp+fn) << "\n";
    cout << "\nSpecificity (Successfully predict cluster 2): " << tn/(fp+tn) << "\n\n";
}

void generateDataTXT(vector<point> data, string filename){
    ofstream ofl;
    ofl.open(filename);
    for(vector<point>::iterator it = data.begin(); it != data.end(); ++it){
        ofl << to_string(it->first) << " " << to_string(it->second) << "\n";
    }
    ofl.close();
}

void plot(){
    ofstream ofl;
    ofl.open("hw4-1plot.gp");
    ofl << "set term qt\n";
    ofl << "set multiplot layout 1,3\n";
    ofl << "set title \"Ground truth\"\n";
    ofl << "plot \"D1.txt\" title \"\" pt 52 lc rgb \"red\", ";
    ofl << "\"D2.txt\" title \"\" pt 52 lc rgb \"blue\"\n";
    ofl << "set title \"Gradient descent\"\n";
    ofl << "plot \"GDpredict0.txt\" title \"\" pt 52 lc rgb \"red\", ";
    ofl << "\"GDpredict1.txt\" title \"\" pt 52 lc rgb \"blue\"\n";
    ofl << "set title \"Newton's method\"\n";
    ofl << "plot \"NMpredict0.txt\" title \"\" pt 52 lc rgb \"red\", ";
    ofl << "\"NMpredict1.txt\" title \"\" pt 52 lc rgb \"blue\"\n";
    ofl.close();

    system("gnuplot -p \"hw4-1plot.gp\"");
}

int main(){
    int N;
    dd mx1, vx1;
    dd my1, vy1;
    dd mx2, vx2;
    dd my2, vy2;
    cout << "input N: ";
    cin >> N;
    cout << "input mean & var of x1: ";
    cin >> mx1 >> vx1;
    cout << "input mean & var of y1: ";
    cin >> my1 >> vy1;
    cout << "input mean & var of x2: ";
    cin >> mx2 >> vx2;
    cout << "input mean & var of y2: ";
    cin >> my2 >> vy2;

    vector<point> D1, D2, inputPoints;
    vector<int> labels;


    int i;

    // generate D1
    for(i = 0; i < N; i++){
        dd x1 = rnorm(mx1, vx1), y1 = rnorm(my1, vy1);
        point p = make_pair(x1, y1);
        D1.push_back(p);
        inputPoints.push_back(p);
        labels.push_back(0);
    }
    generateDataTXT(D1, "D1.txt"); // for plot ground truth

    // generate D2
    for(i = 0; i < N; i++){
        dd x2 = rnorm(mx2, vx2), y2 = rnorm(my2, vy2);
        point p = make_pair(x2, y2);
        D2.push_back(p);
        inputPoints.push_back(p);
        labels.push_back(1);
    }
    generateDataTXT(D2, "D2.txt"); // for plot ground truth

    dd lambda = 2; // learning rate

    // intial w = [0 0 0]^T
    mat w = zeroMatrix(3, 1);
    
    mat X = zeroMatrix(2*N, 3);
    for(i = 0; i < 2*N; i++){
        X[i][0] = 1;
        X[i][1] = inputPoints[i].first;
        X[i][2] = inputPoints[i].second;
    }

    mat Y = zeroMatrix(2*N, 1);
    for(i = 0; i < 2*N; i++){
        if(i < N) Y[i][0] = 0;
        else Y[i][0] = 1;
    }
    cout << "----------------------------------------------\n";
    cout << "Gradient descent: \n\n";
    cout << "w:\n";
    mat w_rlt = GradientDescent(X, w, Y, lambda, 2*N);
    vector<int> predictRlt = predict(X, w_rlt, 2*N); 
    for(i = 0; i < 3; i++){
        cout << fixed << setprecision(10);
        cout << w_rlt[i][0] << "\n";
    }
    pair<vector<point>, vector<point> > sRlt = splitData(inputPoints, predictRlt);
    generateDataTXT(sRlt.first, "GDpredict0.txt");
    generateDataTXT(sRlt.second, "GDpredict1.txt");
    cout << "\nConfusion matrix:\n";
    confusionMatrix(predictRlt, labels, 2*N);
    cout << "----------------------------------------------\n";

    cout << "Newton's method: \n\n";
    w_rlt = NewtonMethod(X, w, Y, 2*N);
    cout << "w:\n";
    predictRlt = predict(X, w_rlt, 2*N);
    for(i = 0; i < 3; i++){
        cout << fixed << setprecision(10);
        cout << w_rlt[i][0] << "\n";
    }
    sRlt = splitData(inputPoints, predictRlt);
    generateDataTXT(sRlt.first, "NMpredict0.txt");
    generateDataTXT(sRlt.second, "NMpredict1.txt");
    cout << "\nConfusion matrix:\n";
    confusionMatrix(predictRlt, labels, 2*N);
    cout << "----------------------------------------------\n";
    plot();
    return 0;
}