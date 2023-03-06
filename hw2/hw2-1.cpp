// hw2-1

#include <iostream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <float.h>

#define ld long double
#define ui unsigned int
#define pi 3.141592653589793

using namespace std;

ui magic;
ui NumTrainingImg, NumTestingImg;
ui NumRows, NumCols;
double discreteModel[10][28*28][32];
double continuousModel[10][28*28][2];
int eachNum[10]; // Traing data中各個數字有幾個
vector<vector<ui> > TrainingImgs;
vector<ui> TrainingLabels;

ui error;

vector<vector<ui> > TestingImgs;
vector<ui> TestingLabels;

double pseudoCount = 1;
double pseudoMean = 128.0;
double pseudoVariance = 5500000;

double dnorm(double value, double mean, double variance){
    double t = 1/(sqrt(2*variance*pi));
    double t1 = (-1)*(value-mean)*(value-mean)/(2*variance);
    double e = exp(t1);
    return t*e;
}

void build_model(int mode){
    int i, j, k;
    // initial model
    for(i = 0; i < 10; i++){
        eachNum[i] = 0;
        for(j = 0; j < 28*28; j++){
            if(mode == 0){
                for(k = 0; k < 32; k++){
                    discreteModel[i][j][k] = pseudoCount;
                }
            }else{
                continuousModel[i][j][0] = pseudoMean; // mean
                continuousModel[i][j][1] = pseudoVariance; // variance
            }
            
        }
    }

    // read training label
    ifstream ifs_1("train-labels-idx1-ubyte", ios::binary);

    // read magic number
    magic = 0;
    for(i = 0; i < 4; i++){
        unsigned char x;
        ifs_1.read((char *)&x, 1);
        unsigned int temp = x;
        magic <<= 8;
        magic += temp;
    }

    // read # of training labels
    NumTrainingImg = 0;
    for(i = 0; i < 4; i++){
        unsigned char x;
        ifs_1.read((char *)&x, 1);
        unsigned int temp = x;
        NumTrainingImg <<= 8;
        NumTrainingImg += temp;
    }

    // read labels
    for(i = 0; i < NumTrainingImg; i++){
        ui tempLabel;
        unsigned char x;
        ifs_1.read((char *)&x, 1);
        tempLabel = x;
        eachNum[tempLabel] += 1;
        TrainingLabels.push_back(tempLabel);
    }
    ifs_1.close();

    // read magic number
    ifstream ifs("train-images-idx3-ubyte", ios::binary);
    magic = 0;
    for(i = 0; i < 4; i++){
        unsigned char x;
        ifs.read((char *)&x, 1);
        unsigned int temp = x;
        magic <<= 8;
        magic += temp;
    }

    // read # of training images
    NumTrainingImg = 0;
    for(i = 0; i < 4; i++){
        unsigned char x;
        ifs.read((char *)&x, 1);
        unsigned int temp = x;
        NumTrainingImg <<= 8;
        NumTrainingImg += temp;
    }
    
    // read # of rows and cols
    NumRows = 0;
    NumCols = 0;
    for(i = 0; i < 4; i++){
        unsigned char x;
        ifs.read((char *)&x, 1);
        unsigned int temp = x;
        NumRows <<= 8;
        NumRows += temp;
    }
    for(i = 0; i < 4; i++){
        unsigned char x;
        ifs.read((char *)&x, 1);
        unsigned int temp = x;
        NumCols <<= 8;
        NumCols += temp;
    }
    
    // read training images, and put into vector
    for(i = 0; i < NumTrainingImg; i++){
        vector<unsigned int> tempImg;
        for(j = 0; j < NumRows*NumCols; j++){
            unsigned char x;
            ifs.read((char *)&x, 1);
            ui t = x;
            tempImg.push_back(t);
        }
        TrainingImgs.push_back(tempImg);
    }
    ifs.close();
                
    if(mode == 0){ // discrete mode
        for(i = 0; i < NumTrainingImg; i++){
            for(j = 0; j < NumRows*NumCols; j++){
                discreteModel[TrainingLabels[i]][j][TrainingImgs[i][j]/8] += 1;
            }
        }
        for(i = 0; i < 10; i++){
            for(j = 0; j < NumRows*NumCols; j++){
                for(k = 0; k < 32; k++){
                    discreteModel[i][j][k] = discreteModel[i][j][k]/double(eachNum[i]);
                }
            }
        }

    }else{ // mode == 1, continueous mode
        // TODO: continueous mode

        // calculate mean for all pixels for each number 
        for(i = 0; i < NumTrainingImg; i++){
            for(j = 0; j < NumRows*NumCols; j++){
                double t = double(TrainingImgs[i][j]);
                continuousModel[TrainingLabels[i]][j][0] += t;
            }
        }
        for(i = 0; i < 10; i++){
            for(j = 0; j < NumRows*NumCols; j++){
                continuousModel[i][j][0] = continuousModel[i][j][0]/double(eachNum[i]);
            }
        }


        for(i = 0; i < NumTrainingImg; i++){
            for(j = 0; j < NumRows*NumCols; j++){
                double t = TrainingImgs[i][j];
                t = t - continuousModel[TrainingLabels[i]][j][0];
                continuousModel[TrainingLabels[i]][j][1] += (t*t);
            }
        }
        for(i = 0; i < 10; i++){
            for(j = 0; j < NumRows*NumCols; j++){
                continuousModel[i][j][1] = continuousModel[i][j][1]/double(eachNum[i]);
            }
        }
    }

}

void predict(int mode){
    error = 0;
    int i, j, k;
    
    // read testing data
    // read testing label
    ifstream ifs_1("t10k-labels-idx1-ubyte", ios::binary);

    // read magic number
    magic = 0;
    for(i = 0; i < 4; i++){
        unsigned char x;
        ifs_1.read((char *)&x, 1);
        unsigned int temp = x;
        magic <<= 8;
        magic += temp;
    }

    // read # of teating labels
    NumTestingImg = 0;
    for(i = 0; i < 4; i++){
        unsigned char x;
        ifs_1.read((char *)&x, 1);
        unsigned int temp = x;
        NumTestingImg <<= 8;
        NumTestingImg += temp;
    }

    // read labels
    for(i = 0; i < NumTestingImg; i++){
        ui tempLabel;
        unsigned char x;
        ifs_1.read((char *)&x, 1);
        tempLabel = x;
        TestingLabels.push_back(tempLabel);
    }
    ifs_1.close();

    // read magic number
    ifstream ifs("t10k-images-idx3-ubyte", ios::binary);
    magic = 0;
    for(i = 0; i < 4; i++){
        unsigned char x;
        ifs.read((char *)&x, 1);
        unsigned int temp = x;
        magic <<= 8;
        magic += temp;
    }

    // read # of testing images
    NumTestingImg = 0;
    for(i = 0; i < 4; i++){
        unsigned char x;
        ifs.read((char *)&x, 1);
        unsigned int temp = x;
        NumTestingImg <<= 8;
        NumTestingImg += temp;
    }
    
    // read # of rows and cols
    NumRows = 0;
    NumCols = 0;
    for(i = 0; i < 4; i++){
        unsigned char x;
        ifs.read((char *)&x, 1);
        unsigned int temp = x;
        NumRows <<= 8;
        NumRows += temp;
    }
    for(i = 0; i < 4; i++){
        unsigned char x;
        ifs.read((char *)&x, 1);
        unsigned int temp = x;
        NumCols <<= 8;
        NumCols += temp;
    }
    
    // read tesing images, and put into vector
    for(i = 0; i < NumTestingImg; i++){
        vector<unsigned int> tempImg;
        for(j = 0; j < NumRows*NumCols; j++){
            unsigned char x;
            ifs.read((char *)&x, 1);
            ui t = x;
            tempImg.push_back(t);
        }
        TestingImgs.push_back(tempImg);
    }
    ifs.close();

    if(mode == 0){ // discrete mode
        double prob[10];
        for(i = 0; i < NumTestingImg; i++){
            for(j = 0; j < 10; j++){
                prob[j] = 0.0;
            }
            double total = 0.0;
            for(j = 0; j < NumRows*NumCols; j++){
                for(k = 0; k < 10; k++){
                    prob[k] -= log10(discreteModel[k][j][TestingImgs[i][j]/8]);
                }
            }
            
            for(k = 0; k < 10; k++){
                double pn = double(eachNum[k])/double(NumTrainingImg);
                prob[k] -= log10(pn);
                total += prob[k];
            }
            cout << "Posterior (in log scale) of Testing image " << i+1 << "\n";
            for(j = 0; j < 10; j++){
                cout << j << ": " << setprecision(15) << prob[j]/total << "\n";
            }
            ui p_num=0;
            double minPosterior = prob[0]/total;
            for(j = 1; j < 10; j++){
                if(minPosterior > prob[j]/total){
                    minPosterior = prob[j]/total;
                    p_num = j;
                }
            }
            // cout << "total: " << setprecision(10) << total << "\n";
            if(p_num != TestingLabels[i]) error += 1;
            cout << "Prediction: " << p_num << ", Ans: " << TestingLabels[i] << "\n\n";
        }

        cout << "\nImagination of numbers in Bayesian classifier:" << "\n" << "\n";
        for(i = 0; i < 10; i++){
            cout << i << ":";
            for(j = 0; j < NumRows*NumCols; j++){
                double cp = 0;
                for(k = 0; k < 16; k++){
                    cp += discreteModel[i][j][k];
                }
                if(!(j % 28)){
                    cout << "\n";
                }
                if(cp < 0.5) {
                    cout << "  1" ;
                }
                else{
                    cout << "  0";
                } 
            }
            cout << "\n";
        }
        cout << "\nError rate: " << double(error)/double(NumTestingImg) << "\n";
    }else{ // mode == 1, continueous mode
        double prob[10];
        for(i = 0; i < NumTestingImg; i++){
            for(j = 0; j < 10; j++){
                prob[j] = 0;
            }
            
            for(j = 0; j < NumRows*NumCols; j++){
                for(k = 0; k < 10; k++){
                    double value = TestingImgs[i][j];
                    double mean = continuousModel[k][j][0];
                    double variance = continuousModel[k][j][1];
                    double p = dnorm(value, mean, variance);
                    prob[k] -= log10(p);
                }
            }
            double total = 0.0;
            for(k = 0; k < 10; k++){
                double pn = double(eachNum[k])/double(NumTrainingImg);
                prob[k] -= log10(pn);
                total += prob[k];
            }

            cout << "Posterior (in log scale) of Testing image " << i+1 << "\n";
            for(j = 0; j < 10; j++){
                cout << j << ": " << setprecision(15) << prob[j]/total << "\n";
            }
            ui p_num=0;
            double minPosterior = prob[0]/total;
            for(j = 1; j < 10; j++){
                if(minPosterior > prob[j]/total){
                    minPosterior = prob[j]/total;
                    p_num = j;
                }
            }
            if(p_num != TestingLabels[i]) error += 1;
            cout << "Prediction: " << p_num << ", Ans: " << TestingLabels[i] << "\n\n";
        }

        cout << "\nImagination of numbers in Bayesian classifier:" << "\n" << "\n";
        for(i = 0; i < 10; i++){
            cout << i << ": ";
            for(j = 0; j < NumRows*NumCols; j++){
                double cp = 0;
                for(k = 0; k < 128; k++){
                    double value = k;
                    double mean = continuousModel[i][j][0];
                    double variance = continuousModel[i][j][1];
                    cp += dnorm(value, mean, variance);
                }
                if(!(j % 28)){
                    cout << "\n";
                }
                if(cp < 0.5){
                    cout << "  1" ;
                }else{
                    cout << "  0";
                }
            }
            cout << "\n";
        }
        cout << "\nError rate: " << double(error)/double(NumTestingImg) << "\n";
    }
}


int main(int argc, char **argv){

    int mode = stoi(argv[1]);
    build_model(mode);
    predict(mode);

    return 0;
}