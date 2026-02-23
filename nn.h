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



class NeuralNetwork
{
    public:
        int inp;
        int hid;
        int out;
        float lr = 0.1; //learning rate

        MatrixXd inputs, outputs, hidden, inputs_T, hidden_T, E_out, E_hid;
        MatrixXd W_IH, W_HO,W_HO_T, W_HO_D, W_IH_D;
        MatrixXd B_H, B_O;

        NeuralNetwork(int inp, int hid, int out, float lr){
            
            this->inp = inp;
            this->hid = hid;
            this->out = out;
            this->lr = lr;
            
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

            hidden = sigmoid(hidden);

            outputs = W_HO*hidden;
            outputs = outputs+B_O;

            outputs = sigmoid(outputs);

            return outputs;
        }

        void train(MatrixXd &inputs, MatrixXd &target){

            // output errors
            outputs = think(inputs);
            E_out = (target - outputs).array();
            E_out = E_out.array() * sigmoid_prime(outputs).array();
        
            // hidden errors
            W_HO_T = W_HO.transpose();
            E_hid = (W_HO_T * E_out).array() * sigmoid_prime(hidden).array();

            // find amounts to adjust weights
            hidden_T = hidden.transpose();
            inputs_T = inputs.transpose();
            // output weights
            W_HO_D = lr * E_out * hidden_T;
            // hidden weights
            W_IH_D = lr * E_hid * inputs_T;

            // Adding the deltas with Weights
            W_HO = W_HO + W_HO_D;
            W_IH = W_IH + W_IH_D;


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
