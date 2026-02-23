#include <nn.h>

int input = 2;
int hidden = 3;
int output = 2;

int main() {
    // Matrix <float, 3,3> mat;

    // mat.setZero();

    // cout<<mat<<endl;

    NeuralNetwork nn(input,hidden,output);

    nn.hidden << 1,2,3;
    nn.inputs << 1,2;
    nn.W_IH << 1,2, 3,4, 5,6;
    nn.W_HO << 1, 2, 3, 4,5,6; 
    MatrixXd target;
    target.resize(output, 1);
    target<< 78,177;

    cout<<"W_HO:\n"<<nn.W_HO<<endl<<"W_IH\n"<<nn.W_IH<<endl;
    cout<<"outputs\n"<<nn.think(nn.inputs)<<endl;
    cout<<"target:\n"<<target<<"\nerrors:\n";
    nn.train(nn.inputs,target);
}
