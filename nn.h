#include <iostream>
#include <lib/Eigen/Dense>
#include <cmath>  
#include <vector>
#include <cstdlib>
#include <ctime>
#include <random>
#include <fstream>
#include <cstdint>
#include <array>
#include <string>
#include <stdexcept>

using namespace std;
using namespace Eigen;

MatrixXd sigmoid(const MatrixXd& Z);
MatrixXd sigmoid_prime(const MatrixXd& Z);
double rand_uniform(double min, double max);
int rand_int(int lo, int hi);

class NeuralNetwork
{
    public:
        int inp;
        int hid;
        int out;
        double lr = 0.1; //learning rate

        MatrixXd inputs, outputs, hidden, target, inputs_T, hidden_T, E_out, E_hid;
        MatrixXd W_IH, W_HO,W_HO_T, W_HO_D, W_IH_D;
        MatrixXd B_H, B_O;

        NeuralNetwork(int inp, int hid, int out, float lr){
            std::srand((unsigned)std::time(nullptr)); 
            this->inp = inp;
            this->hid = hid;
            this->out = out;
            this->lr = lr;
            
            inputs.resize(inp,1);
            outputs.resize(out,1);
            hidden.resize(hid,1);
            target.resize(out,1);


            W_IH.resize(hid, inp);
            W_HO.resize(out, hid);
            
            W_IH = 0.5 * MatrixXd::Random(hid, inp);  
            W_HO = 0.5 * MatrixXd::Random(out, hid);  

            B_O.resize(out, 1);
            B_H.resize(hid, 1);
            B_H = MatrixXd::Zero(hid, 1);
            B_O = MatrixXd::Zero(out, 1);


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
            W_HO += W_HO_D;
            W_IH += W_IH_D;

            // Adjusting biases
            B_O += lr * E_out;
            B_H += lr * E_hid;

        }
};

MatrixXd sigmoid(const MatrixXd& Z) {
    return (1.0 / (1.0 + (-Z.array()).exp())).matrix();
}
MatrixXd sigmoid_prime(const MatrixXd& A) {
    return A.array() * (1.0 - A.array());   // assumes A = sigmoid(Z)
}


int rand_int(int lo, int hi) {
    static thread_local std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<int> dist(lo, hi);  // inclusive [web:365]
    return dist(rng);
}