#include <nn.h>
//#include "data/cat_tr.h"
// #include "data/car_tr.h"
// #include "data/apple_tr.h"
// #include "data/frog_tr.h"
// #include "data/banana_tr.h"

// includes for graphics
#include <iostream>
#include <iomanip>

int input = 784;
int hidden = 128;
int output = 5;
int N = 5000;
uint8_t img[28][28];
int r_int;
double learning_rate = 0.1;

MatrixXd data_inp[5000];
MatrixXd data_t[5000];

void test();
void load();
NeuralNetwork nn(input, hidden, output, learning_rate);

int main() {
  
   
    // load();
    
    test();
    return 0;

}



// void load(){

//     for (int i = 0; i < N; i++) {
//         data_inp[i].resize(28, 28);  
//     }

//     for (int i = 0; i < 1000; i++) {
//         Map<const Matrix<float, 28, 28, RowMajor>> m(&cat_tr[i][0][0]);
//         data_inp[i] = m.cast<double>();
//     }

//     cout<<"data loaded successfully"<<endl;
// }




void test(){

    // tests XOR
    NeuralNetwork nn_t(2, 4, 1,0.1);

    MatrixXd data_inp;
    MatrixXd data_t;

    data_inp.resize(4,2);
    data_t.resize(4,1);
    data_inp << 0,0, 0,1, 1,0, 1,1;
    data_t << 0, 1, 1, 0;

    cout<<"before train"<<endl;
    cout<<"Weights:\nW_IH\n"<<nn_t.W_IH<<"\nW_H\n"<<nn_t.W_HO<<endl; 
    for (int i = 0; i < 1000000; i++)
    {
        r_int = rand_int(0,3);
        nn_t.inputs = data_inp.row(r_int).transpose();
        nn_t.target = data_t.row(r_int).transpose();
        nn_t.train(nn_t.inputs, nn_t.target);
    }
    
    cout<<"after train"<<endl;
    cout<<"Weights:\nW_IH\n"<<nn_t.W_IH<<"\nW_H\n"<<nn_t.W_HO<<endl; 

    cout<<"INFERENCE:\n";
    // for (int i = 0; i < 4; i++)
    // {
    //     nn_t.inputs = data_inp.row(i).transpose();
    //     nn_t.target = data_t.row(i).transpose();
    //     cout<<nn_t.inputs<<"\n\n"<<nn_t.think(nn_t.inputs)<<"\n\n";
        
    // }

    std::cout << "Input 1 | Input 2 | Output\n";
    std::cout << "--------------------------\n";

    for (int i = 0; i < 4; ++i) {
        nn_t.inputs = data_inp.row(i).transpose();
        nn_t.target = data_t.row(i).transpose();
        
       double in1 = nn_t.inputs(0);
        double in2 = nn_t.inputs(1);
        double out = nn_t.think(nn_t.inputs)(0);

        std::cout << std::setw(4) << in1 << "    | " 
                << std::setw(4) << in2 << "    | " 
                << std::setw(4) << out << "\n";
}

}