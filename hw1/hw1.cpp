#include <iostream>
#include <math.h>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <float.h>



#define N 128
#define dd double

using namespace std;

void MatrixMul(dd A[N][N], dd B[N][N], dd C[N][N], int m, int p, int n){
    /*
        A: m by p, B: p by n, C: m by n
        C = AB
    */
    int i, j ,k;
    for(i = 0; i < N; i++){
        for(j = 0; j < N; j++){
            C[i][j] = 0;
        }
    }
    for(i = 0; i < m; i++){
        for(j = 0; j < p; j++){
            for(k = 0; k < n; k++){
                C[i][k] += A[i][j]*B[j][k];
            }
        }
    }

}

void MatrixSub(dd A[N][N], dd B[N][N], dd C[N][N], int m, int n){
    // C = A-B, where A, B, and C are m by n matrix
    int i, j;
    for(i = 0; i < m; i++){
        for(j = 0; j < n; j++){
            C[i][j] = A[i][j] - B[i][j];
        }
    }
}

void TransposeMatrix(dd A[N][N], dd B[N][N], int m, int n){
    /*
        A: m by n, B: n by m
        B = A^T
    */
    int i, j;
    for(i = 0; i < m; i++){
        for(j = 0; j < n; j++){
            B[j][i] = A[i][j];
        }
    }

}

void LUdecomposition(dd A[N][N], dd L[N][N], dd U[N][N], int n){
    int i, j, k;

    /* 
        L = | 1 0 0 |
            | 0 1 0 |
            | 0 0 1 |
        U = A
    */       
    for(i = 0; i < n; i++){
        for(j = 0; j < n; j++){
            if(i == j) {
                L[i][j] = 1;
                U[i][j] = A[i][j];
                continue;
            }
            L[i][j] = 0;
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


}
void inverseU(dd U[N][N], dd iU[N][N], int m){
    int i, j, k;
    dd tmp[m][m];
    // iU = I, tmp = U;
    for(i = 0; i < m; i++){
        for(j = 0; j < m; j++){
            if(i == j){
                iU[i][j] = 1.0;
            }else{
                iU[i][j] = 0;
            }
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
}

void inverseL(dd L[N][N], dd iL[N][N], int m){
    int i, j, k;
    dd tmp[m][m];
    // iU = I, tmp = L;
    for(i = 0; i < m; i++){
        for(j = 0; j < m; j++){
            if(i == j){
                iL[i][j] = 1.0;
            }else{
                iL[i][j] = 0;
            }
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
}

void InverseMatrix(dd A[N][N], dd iA[N][N], int m){
    // iA = A^-1
    int i, j;
    dd L[N][N], U[N][N], iL[N][N], iU[N][N];
    LUdecomposition(A, L, U, m); // A = LU
    inverseL(L, iL, m); // iL = L^-1
    inverseU(U, iU, m); // iU = U^-1
    MatrixMul(iU, iL, iA, m, m, m); // iA = U^-1 * L^-1
}

// LSE
void LSE(dd A[N][N], dd x[N][N], dd b[N][N], int len, int n, int lambda){
    int i, j;
    dd At[N][N]; // A^T
    dd AtA[N][N]; // A^T * A
    dd iA[N][N]; // inverse of A^T*A
    dd B[N][N]; // B = (A^T*A)^-1*A^T
    TransposeMatrix(A, At, len, n); // At = A^T
    MatrixMul(At, A, AtA, n, len, n); // AtA = A^T*A

    // AtA = A^T*A + lambda*I
    for(i = 0; i < n; i++){
        AtA[i][i] = AtA[i][i]+lambda;
    }
    
    InverseMatrix(AtA, iA, n); //iA = (AtA)^-1
    MatrixMul(iA, At, B, n, n, len); // B = (A^T*A)^-1*A^T
    MatrixMul(B, b, x, n, len, 1); // x = (A^T*A)^-1*A^T*b
}

//Newton's method
void NewtonMethod(dd A[N][N], dd xn[N][N], dd b[N][N], dd xn_1[N][N], int len,int n){
    // df: gradient function, Hf: Hessian Matrix
    /*
    df = 2A^T*Ax-2A^Tb = 2(A^T*Ax-A^Tb)
    Hf = 2A^T*A
    xn_1 = xn-Hf^-1*df
    */
    int i, j;
    dd df[N][N], Hf[N][N];
    dd At[N][N];
    TransposeMatrix(A, At, len, n); // At = A^T
    dd AtA[N][N];
    MatrixMul(At, A, AtA, n, len, n); // AtA = A^T*A
    dd Atb[N][N];
    MatrixMul(At, b, Atb, n, len, 1); // Atb = A^T*b
    dd AtAx[N][N];
    MatrixMul(AtA, xn, AtAx, n, n, 1); // AtAx = A^T*A*x
    
    // df = 2A^T*Ax-2A^Tb
    MatrixSub(AtAx, Atb, df, n, 1);
    for(i =0; i < n; i++){
        df[i][0] = 2*df[i][0];
    }

    // Hf = 2A^t*A
    for(i = 0; i < n; i++){
        for(j = 0; j < n; j++){
            Hf[i][j] = 2*AtA[i][j];
        }
    }

    InverseMatrix(Hf, At, n); // At = Hf^-1
    MatrixMul(At, df, AtA, n, n, 1); // AtA = Hf^-1 * df
    MatrixSub(xn, AtA, xn_1, n, 1); // xn_1 = xn - Hf^-1 * df
}

void plot(string formula1, string formula2, string datafile, dd min_x, dd max_x){
    ofstream ofile;
    ofile.open("PlotChart.gp");
    ofile << "set term qt\n";
    ofile << "set multiplot layout 2,1\n";
    ofile << "set title \"LSE\"\n";
    ofile << "plot [" << to_string(min_x) << ":" << to_string(max_x) << "] "<< formula1 << " lt rgb \"red\" lw 1.2,";
    ofile << "\"" << datafile << "\" using 1:2 title \"Data point\" pt 52 ps 0.7 lc rgb \"blue\"\n";
    ofile << "set title \"Newton's method\"\n";
    ofile << "plot [" << to_string(min_x) << ":" << to_string(max_x) << "] " << formula2 << " lt rgb \"red\" lw 1.2,";
    ofile << "\"" << datafile << "\" using 1:2 title \"Data point\" pt 52 ps 0.7 lc rgb \"blue\"\n";
    ofile.close();

    system("gnuplot -p \"PlotChart.gp\"");
}

int main(int argc, char **argv){
    int n, lambda;
    int i, j;
    n = stoi(argv[2]);
    lambda = stoi(argv[3]);
    cout << "n = " << n << ", lambda = " << lambda << endl;

    // read the input file
    ifstream ifs(string(argv[1]), ios::in);
    vector<dd> x;
    vector<dd> y;
    dd min_x = DBL_MAX, max_x = DBL_MIN;
    if(!ifs.is_open()){
        cout << "Error to open file" << endl;
    }else{
        string s, t;
        dd t1, t2;
        while(getline(ifs, s, ',') && getline(ifs, t)){
            if(stod(s) < min_x) min_x = stod(s);
            if(stod(s) > max_x) max_x = stod(s);
            x.push_back(stod(s));
            y.push_back(stod(t));
        }
        ifs.close();
    }
    ifs.close();

    if(min_x <= 0) min_x = 1.3*min_x;
    else min_x= -0.3*min_x;
    max_x = 1.3*max_x;

    // write a datafile used to plot chart in gnuplot
    ofstream ofile;
    ofile.open("dataFile.txt");
    for(i = 0; i < x.size(); i++){
        ofile << to_string(x.at(i)) << " " << to_string(y.at(i)) << "\n";
    }
    ofile.close();

    int len = x.size();
    dd A[N][N];
    dd b[N][N];
    dd xx[N][N];

    for(i = 0; i < len; i++){
        for(j = n-1; j >= 0; j--){
            A[i][j] = pow(x.at(i), n-j-1);
        }
        b[i][0] = y.at(i);
    }


    // ---------- LSE ----------------
    cout << "LSE: " << endl;
    LSE(A, xx, b, len, n, lambda);
    
    //formula is for print on screen, and plotLSE_F is for plot
    string formula = "", plotLSE_F = ""; 
    for(i = 0; i < n-2; i++){
        formula = formula + to_string(xx[i][0]) + "*X^" + to_string(n-1-i) + " + ";
        plotLSE_F = plotLSE_F + to_string(xx[i][0]) + "*x**" + to_string(n-1-i) + " + ";
    }
    formula = formula + to_string(xx[n-2][0]) + "*X + " + to_string(xx[n-1][0]);
    plotLSE_F = plotLSE_F + to_string(xx[n-2][0]) + "*x + " + to_string(xx[n-1][0]);
    cout << "Fitting line: " << formula << endl;
    
    // compute total error
    // Error(f(x), y) = sum((f(x_i)-y_i)^2)
    cout << "Total error = " ;
    dd te = 0;
    for(i = 0; i < len; i++){
        dd t = 0;
        for(j = n-1; j >= 0; j--){
            t += xx[n-1-j][0]*pow(x.at(i), j);
        }
        te += pow(t-y.at(i),2);
    }
    cout << te << endl; 

    // ---------- Newton's method ----------------
    cout << "Newton's Method: " << endl;
    dd xn[N][N]; // initial point [0, ..., 0]^T
    dd xn_1[N][N];
    for(i = 0; i < n; i++){
        xn[i][0] = 0;
    }
    NewtonMethod(A, xn, b, xn_1, len, n);

    //formula is for print on screen, and plotNM_F is for plot
    formula = "";
    string plotNM_F = "";
    for(i = 0; i < n-2; i++){
        formula = formula + to_string(xn_1[i][0]) + "*X^" + to_string(n-1-i) + " + ";
        plotNM_F = plotNM_F + to_string(xn_1[i][0]) + "*x**" + to_string(n-1-i) + " + ";
    }
    formula = formula + to_string(xn_1[n-2][0]) + "*X + " + to_string(xn_1[n-1][0]);
    plotNM_F = plotNM_F + to_string(xn_1[n-2][0]) + "*x + " + to_string(xn_1[n-1][0]);
    cout << "Fitting line: " << formula << endl;
    
    // compute total error
    // Error(f(x), y) = sum((f(x_i)-y_i)^2)
    cout << "Total error = " ;
    te = 0;
    for(i = 0; i < len; i++){
        dd t = 0;
        for(j = n-1; j >= 0; j--){
            t += xn_1[n-1-j][0]*pow(x.at(i), j);
        }
        te += pow(t-y.at(i),2);
    }
    cout << te << endl;

    //plot chart for each fitting line and data point
    plot(plotLSE_F, plotNM_F,  "dataFile.txt", min_x, max_x);

    // delete the file we created
    // int status;
    // status = remove("dataFile.txt");
    // status = remove("PlotChart.gp");
    return 0;
}