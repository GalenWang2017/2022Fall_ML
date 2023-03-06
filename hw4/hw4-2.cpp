// hw 4-2
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
#define ui unsigned int
#define pi 3.141592653589793

using namespace std;

ui magic;
ui NumTrainingImg;
ui NumRows, NumCols;
vector<vector<ui> > imgs; // all training images
vector<ui> labels; // label of each images
vector<vector<dd> > pixelProb; // the probability of each pixels for each digit, each pixel ~ Bernoulli(p)
vector<dd> digitProb; // the probability of each digits
vector<vector<dd> > W; // the probability of each images is digit i or not, i = 0~9
int numIterations = 1;
ui classLabel[10] = {0}; // Map a label for a class

default_random_engine generator(time(NULL));
uniform_real_distribution<dd> distribution(0.2, 0.8);

void readImages(){
    int i, j;
    // read training images

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
            if(t > 127) tempImg.push_back(1);
            else tempImg.push_back(0);
        }
        imgs.push_back(tempImg);
    }
    ifs.close();
}

void readLabels(){
    // read training label
    int i;

    ifstream ifile("train-labels-idx1-ubyte", ios::binary);

    // read magic number
    magic = 0;
    for(i = 0; i < 4; i++){
        unsigned char x;
        ifile.read((char *)&x, 1);
        unsigned int temp = x;
        magic <<= 8;
        magic += temp;
    }

    // read # of training labels
    NumTrainingImg = 0;
    for(i = 0; i < 4; i++){
        unsigned char x;
        ifile.read((char *)&x, 1);
        unsigned int temp = x;
        NumTrainingImg <<= 8;
        NumTrainingImg += temp;
    }

    // read labels
    for(i = 0; i < NumTrainingImg; i++){
        ui tempLabel;
        unsigned char x;
        ifile.read((char *)&x, 1);
        tempLabel = x;
        labels.push_back(tempLabel);
    }
    ifile.close();
}

void initialParameters(){
    // initial digitProb and pixelProb
    pixelProb.clear();
    digitProb.clear();
    W.clear();
    int i, j;
    for(i = 0; i < 10; i++){
        vector<dd> t;
        for(j = 0; j < NumRows*NumCols; j++){
            dd p = distribution(generator);
            t.push_back(p);
        }
        pixelProb.push_back(t);

        dd dprob = distribution(generator);
        digitProb.push_back(dprob);
    }

    // initial W
    for(i = 0; i < NumTrainingImg; i++){
        vector<dd> t;
        for(j = 0; j < 10; j++){
            t.push_back(0);
        }
        W.push_back(t);
    }
}

void EMalgo(){
    int i, j, k;
    dd difference = 0.0;
    while(1){
        difference = 0.0;
        // E step
        /*
            compute W
        */
        for(i = 0; i < NumTrainingImg; i++){
            dd temp = 0;
            for(j = 0; j < 10; j++){
                W[i][j] = 1;
                for(k = 0; k < NumRows*NumCols; k++){
                    if(imgs[i][k] == 1){
                        W[i][j] *= pixelProb[j][k];
                    }else{
                        W[i][j] *= (1-pixelProb[j][k]);
                    }
                }
                W[i][j] *= digitProb[j];
                temp += W[i][j];
            }
            if(temp != 0){
                for(j = 0; j < 10; j++){
                    W[i][j] /= temp;
                }
            }
        }

        // M step
        /*
            compute pixelProb and digitProb
        */
        vector<dd> prev_digitProb = digitProb;
        for(i = 0; i < 10; i++){
            dd new_digitProb = 0;
            for(j = 0; j < NumTrainingImg; j++){
                new_digitProb += W[j][i];
            }

            for(j = 0; j < NumRows*NumCols; j++){
                dd new_pixelProb = 0;
                for(k = 0; k < NumTrainingImg; k++){
                    new_pixelProb += (W[k][i]*imgs[k][j]);
                }
                new_pixelProb = (new_pixelProb)/(new_digitProb);
                if(!new_pixelProb) new_pixelProb = 1e-8;
                difference += (new_pixelProb - pixelProb[i][j])*(new_pixelProb - pixelProb[i][j]);
                pixelProb[i][j] = new_pixelProb;
            }
            difference += (new_digitProb/NumTrainingImg-digitProb[i])*(new_digitProb/NumTrainingImg-digitProb[i]);
            digitProb[i] = new_digitProb/NumTrainingImg;
        }

        for(i = 0; i < 10; i++){
            cout << "class " << i << ":\n";
            for(j = 0; j < NumRows*NumCols; j++){
                if(pixelProb[i][j] >= 0.5) cout << "1 ";
                else cout << "0 ";
                if(!((j+1)%NumRows)) cout << "\n";
            }
            cout << "\n";
        }
        cout << "\n";
        cout << fixed << setprecision(10);
        cout << "No. of iteration: " << numIterations << ", Difference: " << difference << "\n\n";
        if(numIterations == 15 || difference <= 5e-1) break;
        numIterations += 1;
    }
    
}

void assignClassToLabel(){
    ui i, j, k;
    vector<vector<ui> > eachClass; 
    for(i = 0; i < 10; i++){
        vector<ui> c;
        eachClass.push_back(c);
    }
    for(i = 0; i < NumTrainingImg; i++){
        for(j = 0; j < 10; j++){
            if(W[i][j] >= 0.5) eachClass[j].push_back(i);
        }
    }
    for(k = 0; k < 10; k++){
        ui labelCount[10] = {0}; // count number of label in this class k
        for(vector<ui>::iterator it = eachClass[k].begin(); it != eachClass[k].end(); it++){
            labelCount[labels[*it]] += 1;
        }
        ui maxCount = 0;
        for(j = 0; j < 10; j++){
            if(labelCount[j] > maxCount){
                maxCount = labelCount[j];
                classLabel[k] = j;
            }
        }
    }
    vector<ui> redundantIndex;
    vector<ui> restLabel;
    for(k = 0; k < 10; k++){
        ui c = 0;
        for(j = 0; j < 10; j++){
            if(classLabel[j] == k) {
                c+= 1;
                if(c >= 2) redundantIndex.push_back(j);
            }
        }
        if(c == 0) restLabel.push_back(k); 
    }
    while (!redundantIndex.empty()){
        classLabel[redundantIndex.back()] = restLabel.back();
        redundantIndex.pop_back();
        restLabel.pop_back();
        if(restLabel.empty()) break;
    }
}

void printLabeledClass(){
    int i, j;
    for(i = 0; i < 10; i++){
        ui cl = 0;
        for(j = 0; j < 10; j++){
            if(classLabel[j] == i){
                cl = j;
                break;
            }
        }
        cout << "Labeled class " << i << "\n";
        for(j = 0; j < NumRows*NumCols; j++){
            if(pixelProb[cl][j] >= 0.5){
                cout << "1 ";
            }else{
                cout << "0 ";
            }
            if(!((j+1)%NumRows)) cout << "\n";
        }
        cout << "\n";
    }
}

void predict(){
    dd errorPredict = 0;
    int i, j;
    for(i = 0; i < NumTrainingImg; i++){
        ui maxClass = 0;
        dd MaxW = -0.000001;
        for(j = 0; j < 10; j++){
            if(W[i][j] > MaxW){
                MaxW = W[i][j];
                maxClass = j;
            }
        }
        if(classLabel[maxClass] != labels[i]) errorPredict += 1;
    }
    cout << "\nTotal iteration to converge: " << numIterations << "\n";
    cout << "Total error rate: " << fixed << setprecision(10) << errorPredict/double(NumTrainingImg) << "\n";
}

void confusionMatrix(){
    // compute and print confusion matrix
    int i, j;
    for(j = 0; j < 10; j++){
        ui clabel = 0;
        for(i = 0; i < 10; i++){
            if(classLabel[i] == j){
                clabel = i;
                break;
            }
        }
        cout << "-----------------------------------------------------------\n";
        cout << "Confusion matrix " << j << "\n";
        dd tp = 0, tn = 0, fp = 0, fn = 0;
        for(i = 0; i < NumTrainingImg; i++){
            if(W[i][clabel] >= 0.5){ // predict image i is digit j
                if(labels[i] == j){ // label of image i is digit j
                    tp += 1;
                }else{ // label of image i isn't digit j
                    fp += 1;
                }
            }else{ // predict image i isn't digit j
                if(labels[i] != j){ // label og image i isn't digit j
                    tn += 1;
                }else{ // label og image i is digit j
                    fn += 1;
                }
            }
        }
        cout << setprecision(0);
        cout << "\t\t  Predict number "<< j << "\t Predict not number " << j << "\n";
        cout << "   Is number " << j << "\t\t" << tp << "\t\t\t" << fn << "\n"; 
        cout << "Isn't number " << j<< "\t\t" << fp << "\t\t\t" << tn << "\n";
        cout << setprecision(6); 
        cout << "\nSensitivity (Successfully predict number " << j << "): " << tp/(tp+fn) << "\n";
        cout << "\nSpecificity (Successfully predict not number " << j << "): " << tn/(fp+tn) << "\n\n";
    }
    cout << "-----------------------------------------------------------\n";
}

int main(){
    readImages();
    readLabels();

    initialParameters();

    EMalgo();
    cout << "\n\n";
    assignClassToLabel();
    cout << "-----------------------------------------------------------\n";
    cout << "-----------------------------------------------------------\n";
    printLabeledClass();
    confusionMatrix();
    predict();
    return 0;
}