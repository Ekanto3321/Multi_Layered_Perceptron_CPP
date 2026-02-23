#include <iostream>
#include <lib/Eigen/Dense>
#include <cmath>  
#include <vector>

using namespace std;
using namespace Eigen;

MatrixXd sigmoid(const MatrixXd& Z);
MatrixXd sigmoid_prime(const MatrixXd& Z);
MatrixXd relu(const MatrixXd& Z);
MatrixXd relu_prime(const MatrixXd& Z);

float learning_rate = 0.1;


class NeuralNetwork
{
    public:
        int inp;
        int hid;
        int out;

        MatrixXd inputs, outputs, hidden, E_out, E_hid;
        MatrixXd W_IH, W_HO,W_HO_T;
        MatrixXd B_H, B_O;

        NeuralNetwork(int inp, int hid, int out){
            
            this->inp = inp;
            this->hid = hid;
            this->out = out;
            
            inputs.resize(inp,1);
            outputs.resize(out,1);
            hidden.resize(hid,1);

            W_IH.resize(hid, inp);
            W_HO.resize(out, hid);
            
            B_O.resize(out, 1);
            B_H.resize(hid, 1);

        }

        MatrixXd think(MatrixXd &inputs){
            
            hidden = W_IH*inputs;
            hidden = hidden+B_H;

            hidden = relu(hidden);

            outputs = W_HO*hidden;
            outputs = outputs+B_O;

            outputs = relu(outputs);

            return outputs;
        }

        void train(MatrixXd &inputs, MatrixXd &target){

            outputs = think(inputs);
            E_out = (target-outputs).array().square();
            // E_out = (target-outputs);
            
            W_HO_T = W_HO.transpose();

            E_hid = W_HO_T * E_out;

            cout<<E_out<<endl;

        }
};

MatrixXd relu(const MatrixXd& Z) {
    return Z.array().max(0.0);
}
MatrixXd relu_prime(const MatrixXd& Z) {
    return (Z.array() > 0.0).cast<double>();
}

MatrixXd sigmoid(const MatrixXd& Z) {
    return Z.array().unaryExpr([](double x) {
        if (x > 45.0) return 1.0;
        if (x < -45.0) return 0.0;
        return 1.0 / (1.0 + exp(-x));  
    });
}

MatrixXd sigmoid_prime(const MatrixXd& Z) {
    MatrixXd S = sigmoid(Z);
    return S.array() * (1.0 - S.array());
}
