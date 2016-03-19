#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <stdlib.h>
#include <math.h>

using namespace std;

//Globals for the data sets
double train[33][1608];
double test[33][540];
double weights1[5][30]; //weioghts for first layer (5 neurons, 30 inputs)
double weights2[5][5]; //weioghts for second layer (5 neurons, 5 inputs)
double weights3[5]; //weioghts for final layer (one neuron, 5 inputs)
double net1[5]; //raw net from first layer
double net2[5]; //raw net from second layer
double net3; //raw net of final neuron
double out1[5]; //outputs from first layer
double out2[5]; //output from second layer
double out3; //output of final neuron
double delta1[5];
double delta2[5];
double delta3;
double cutoffs[3][201]; //storage for true positive and false positive values at various cutoffs to calculate ROC

void zeroNOD(){
  for(int i = 0; i<5; i++){
    net1[i] = 0;
    net2[i] = 0;
    out1[i] = 0;
    out2[i] = 0;
    delta1[i] = 0;
    delta2[i] = 0;
  }
  net3 = 0;
  out3 = 0;
  delta3 = 0;
}

void dump_weights(){
  ofstream weight_out;
  weight_out.open("weight.txt");
  for(int x = 0; x < 5; x++){
    weight_out << weights3[x] << endl;
    for(int i = 0; i < 30; i++){
      weight_out << weights1[x][i] << endl;
      if(i < 5) weight_out << weights2[x][i] <<endl;
    }
  }
}

void dump_cutoffs(){
  ofstream cutd;
  cutd.open("cutoffs.txt");
  for(int y = 0; y < 201; y++){
    cutd << cutoffs[0][y] << ",";
    cutd << cutoffs[1][y] << ",";
    cutd << cutoffs[2][y] << endl;
  }
}

//reads the data in from the files they are saved in
void readfile(){

  ifstream infileTrain;
  ifstream infileTest;
  char cell[256];
  int y=0;
  infileTrain.open ("training.txt");
  infileTest.open ("test.txt");

  if (infileTrain.is_open() && infileTest.is_open()){ //if they both open successfully
    while (!infileTrain.eof() || !infileTest.eof()){  //while one of them is still going
      for(int x=0; x<33; x++){                        //step through a whole row
        if(!infileTrain.eof()){                       //only pull data if the file isn't over
          if(x == 32) train[x][y] = -1;               //this is our class column, -1 will mean unvisited/unclassified point
          else{
            infileTrain.getline(cell, 256, ',');      //pull a cell from file
            train[x][y] = atof(cell);                   //put the cell into the array
          }
        }
        if(!infileTest.eof()){                        //only pull data if the file isn't over
          if(x == 32) test[x][y] = -1;                //this is our class column, -1 will mean unvisited/unclassified point
          else{
            infileTest.getline(cell, 256, ',');        //pull a cell from file
            test[x][y] = atof(cell);                   //put the cell into the array
          }
        }
      }
        y++ ; //next line!
    }
    infileTrain.close();
    infileTest.close();
  }
  else cerr << "Error opening file" << endl;
}

//set up weights with random small numbers
void init_weights(){
  for(int x = 0; x < 5; x++){
    weights3[x] = (rand() % 99) / 100.0 + 0.01;
    for(int i = 0; i < 30; i++){
      weights1[x][i] = (rand() % 99) / 100.0 + 0.01;
      if(i < 5) weights2[x][i] = (rand() % 99) / 100.0 + 0.01;
    }
  }
}

//take a neurons net and calculate out
double net2out(double x){
  //double y = (exp(x) - exp(-x)) / (exp(x) + exp(-x)); //hyperbolic tangent
  double y = 1 / (1+exp(-x)); //sigmoidal
  return y;
}

double fprime(double x){
  //double y = 2.0 / ((exp(x) + exp(-x)) * (exp(x) + exp(-x))); //hyperbolic tangent
  double y = net2out(x) * (1 - net2out(x)); //sigmoidal
  return y;
}

//perceptron learning algorithm with delta scores and learning coefficient
void learn(){
  int subepoch = 1;
  double prev_error = 0;
  double error = 0;
  double error_dif = 0;
  double LC = 0.1;
  double epoch = 1;
  while(epoch <= 25){
    error = 0;
    for(int y = 0; y < 1608; y++){ //for each patient, run forwards, then back prop, repeat as needed
      int subepoch = 1;
      while(subepoch <= 5){
        subepoch++; //error is acting weird, control for while loop until fixed
        zeroNOD(); //reset all net, out, and delta arrays to 0

        //FORWARD PASS

        //first layer into net1 then out1
        for(int neuron = 0; neuron < 5; neuron++){
          for(int input = 1; input <= 30; input++){
            net1[neuron] += train[input][y] * weights1[neuron][input];
          }
        }
        for(int i = 0; i < 5; i++){
          out1[i] = net2out(net1[i]);
        }
        //second layer into net2 then out2
        for(int neuron = 0; neuron < 5; neuron++){
          for(int input = 0; input < 5; input++){
            net2[neuron] += out1[input] * weights2[neuron][input];
          }
        }
        for(int i = 0; i < 5; i++){
          out2[i] = net2out(net2[i]);
        }
        //last layer into net3 then out3
        for(int input = 0; input < 5; input++){
          net3 += out2[input] * weights3[input];
        }
        out3 = net2out(net3);


        //BACK PROP TIME!

        //delta for final layer
        delta3 = (train[31][y] - out3)*(fprime(net3));
        //final layer weight adjust
        for(int i = 0; i < 5; i++){
          weights3[i] += LC * delta3 * out2[i];
        }

        //delta for middle layer
        for(int d = 0; d < 5; d++){
          delta2[d] = delta3 * weights3[d] * fprime(net2[d]);
        }
        //middle layer weight adjust
        for(int l2 = 0; l2 < 5; l2++){
          for(int l1 = 0; l1 < 5; l1++){
            weights2[l2][l1] += LC * delta2[l2] * out1[l1];
          }
        }

        //delta for first layer
        for(int d = 0; d < 5; d++){
          for(int n = 0; n < 5; n++){
            delta1[d] += delta2[n] * weights2[d][n];
          }
            delta1[d] *= fprime(net1[d]);
        }
        //weight adjust for first layer
        for(int l1 = 0; l1 < 5; l1++){
          for(int in = 1; in <= 30; in++){
            weights1[l1][in] += LC * delta1[l1] * train[in][y];
          }
        }

        //error resolution, decide if error was more last time
        if(y == 0) error_dif = 1;
        else error_dif = prev_error - delta3;
        prev_error = delta3;
      }
      error += delta3*delta3; //accumulate the error^2 and divide by n after the for loop is up
    }
    error = error/1608; //divide by n after for loop
    epoch++;
    cout << "Error = " << error << endl;
  }
  cout << "Some weights = " << weights1[0][0] << ", " << weights1[1][1] << ", " << weights2[0][0] << ", " << weights2[1][1] << ", " << weights3[0] << endl;
  cout << "Final Error = " << error << endl;
}

//use classifier to determine each points class
//simply applies the formula with the generate weights from learn()
void classify_train(){
  for(int y = 0; y < 1608; y++){
    zeroNOD();
    //first layer into net1 then out1
    for(int neuron = 0; neuron < 5; neuron++){
      for(int input = 1; input <= 30; input++){
        net1[neuron] += train[input][y] * weights1[neuron][input];
      }
    }
    for(int i = 0; i < 5; i++){
      out1[i] = net2out(net1[i]);
    }

    //second layer into net2 then out2
    for(int neuron = 0; neuron < 5; neuron++){
      for(int input = 0; input < 5; input++){
        net2[neuron] += out1[input] * weights2[neuron][input];
      }
    }
    for(int i = 0; i < 5; i++){
      out2[i] = net2out(net2[i]);
    }
    //last layer into net3 then out3
    for(int input = 0; input < 5; input++){
      net3 += out2[input] * weights3[input];
    }
    out3 = net2out(net3);
    train[32][y] = out3;
  }

}

void classify_test(){
  for(int y = 0; y < 540; y++){
    zeroNOD();
    //first layer into net1 then out1
    for(int neuron = 0; neuron < 5; neuron++){
      for(int input = 1; input <= 30; input++){
        net1[neuron] += test[input][y] * weights1[neuron][input];
      }
    }
    for(int i = 0; i < 5; i++){
      out1[i] = net2out(net1[i]);
    }
    //second layer into net2 then out2
    for(int neuron = 0; neuron < 5; neuron++){
      for(int input = 0; input < 5; input++){
        net2[neuron] += out1[input] * weights2[neuron][input];
      }
    }
    for(int i = 0; i < 5; i++){
      out2[i] = net2out(net2[i]);
    }
    //last layer into net3 then out3
    for(int input = 0; input < 5; input++){
      net3 += out2[input] * weights3[input];
    }
    out3 = net2out(net3);
    test[32][y] = out3;
  }
}

//using incremental cutoffs, determine true positive and false positive for each
void ROC_train(){
  double cut = 1.0;
  for(int i = 0; i < 201; i++){
    double TP = 0;
    double FP = 0;
    double total = 0;
    for(int y = 0; y < 1608; y++){
      if(train[31][y] == 1) total++; //count positives
      if(train[32][y] > cut){ //if the NN called positive
        if(train[31][y] == 1) TP++; //it was right
        if(train[31][y] == 0) FP++; //it was wrong
      }
    }
    cutoffs[0][i] = cut;
    cutoffs[1][i] = TP / total; //percent right out of positives
    cutoffs[2][i] = FP / (1608-total); //percent right out of negatives
    cut -= 0.005;
    if(cut < 0) cut = 0;
  }
}

void ROC_test(){
  double cut = 1.0;
  for(int i = 0; i < 201; i++){
    double TP = 0;
    double FP = 0;
    double total = 0;
    for(int y = 0; y < 540; y++){
      if(test[31][y] == 1) total++; //count positives
      if(test[32][y] > cut){ //if the NN called positive
        if(test[31][y] == 1) TP++; //it was right
        if(test[31][y] == 0) FP++; //it was wrong
      }
    }
    cutoffs[0][i] = cut;
    cutoffs[1][i] = TP / total; //percent right out of positives
    cutoffs[2][i] = FP / (540-total); //percent right out of negatives
    cut -= 0.005;
    if(cut < 0) cut = 0;
  }
}

//analize data for results of classification and generate area under the curve of ROC
void TRAINresults(){
  double AUC = 0;
  ROC_train();
  cout << "middle cutoff for train (cut, TP, FP): " << cutoffs[0][100] << ", " << cutoffs[1][100] << ", " << cutoffs[2][100] << endl;
  for(int i = 0; i < 200; i++){
    AUC += cutoffs[1][i] * (cutoffs[2][i+1] - cutoffs[2][i]); //add the rectangle
    AUC += (cutoffs[1][i+1]-cutoffs[1][i]) * (cutoffs[2][i+1] - cutoffs[2][i]) / 2.0; //add the triangle
  }
  cout << "Area Under Curve of Training set = " << AUC << endl;
}

void TESTresults(){
  double AUC = 0;
  ROC_test();
  cout << "middle cutoff for test (cut, TP, FP): " << cutoffs[0][100] << ", " << cutoffs[1][100] << ", " << cutoffs[2][100] << endl;
  for(int i = 0; i < 200; i++){
    AUC += cutoffs[1][i] * (cutoffs[2][i+1] - cutoffs[2][i]);
    AUC += (cutoffs[1][i+1]-cutoffs[1][i]) * (cutoffs[2][i+1] - cutoffs[2][i]) / 2.0;
  }
  cout << "Area Under Curve of Test set = " << AUC << endl;
}

int main(){
  readfile(); //read data into global arrays
  init_weights(); //random numbers between .01 and 1 for the weights

  learn(); //neural net learning
  dump_weights();

  classify_train(); //classification based on weights genereated by learn()
  TRAINresults(); //output ROC area under curve results of training set

  classify_test();
  TESTresults();

  dump_cutoffs();
  return 0;
}
