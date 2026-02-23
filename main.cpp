#include <nn.h>

int input = 2;
int hidden = 3;
int output = 2;
int learning_rate = 0.1;

int main() {


    NeuralNetwork nn(input,hidden,output, learning_rate);

    nn.hidden << 1,2,3;
    nn.inputs << 1,2;
    nn.W_IH << 1,2, 3,4, 5,6;
    nn.W_HO << 1, 2, 3, 4,5,6; 
    MatrixXd target;
    target.resize(output, 1);
    target<< 1,1;

    cout<<"W_HO:\n"<<nn.W_HO<<endl<<"W_IH\n"<<nn.W_IH<<endl;
    cout<<"outputs\n"<<nn.think(nn.inputs)<<endl;
    cout<<"target:\n"<<target<<"\nerrors:\n";
    nn.train(nn.inputs,target);
}
