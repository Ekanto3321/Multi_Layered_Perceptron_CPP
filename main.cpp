#include <nn.h>

int input = 2;
int hidden = 2;
int output = 1;
int r_int;
double learning_rate = 0.1;

MatrixXd data_inp;
MatrixXd data_t;

int main() {


  
    NeuralNetwork nn(input, hidden, output, learning_rate);
    
    data_inp.resize(4,2);
    data_t.resize(4,1);
    data_inp << 0,0, 0,1, 1,0, 1,1;
    data_t << 0, 1, 1, 0;


    // nn.inputs = data_inp.row(0).transpose();


    // // nn.hidden << 1,2,3;
    // nn.inputs << 1,2;
    // MatrixXd target;
    // target.resize(output, 1);
    // target<< 1;



    cout<<"before train"<<endl;
    cout<<"Weights:\nW_IH\n"<<nn.W_IH<<"\nW_H\n"<<nn.W_HO<<endl; 
    for (int i = 0; i < 100000; i++)
    {
        r_int = rand_int(0,3);
        nn.inputs = data_inp.row(r_int).transpose();
        nn.target = data_t.row(r_int).transpose();
        nn.train(nn.inputs, nn.target);
    }
    
    cout<<"after train"<<endl;
    cout<<"Weights:\nW_IH\n"<<nn.W_IH<<"\nW_H\n"<<nn.W_HO<<endl; 

    cout<<"INFERENCE:\n";
    for (int i = 0; i < 4; i++)
    {
        nn.inputs = data_inp.row(i).transpose();
        nn.target = data_t.row(i).transpose();
        cout<<nn.inputs<<"\n\n"<<nn.think(nn.inputs)<<"\n\n";
        
    }

}
