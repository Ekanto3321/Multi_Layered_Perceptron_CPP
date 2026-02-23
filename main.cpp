#include <nn.h>

int input = 784;
int hidden = 128;
int output = 5;
uint8_t img[28][28];
int r_int;
double learning_rate = 0.1;

MatrixXd data_inp[5000];
MatrixXd data_t[5000];

void test();
NeuralNetwork nn(input, hidden, output, learning_rate);

int main() {
  
   
    load();
    
    
    
    // test();
    return 0;

}

void load(){

    for (int i = 0; i < 1000; i++)
    {
        /* code */
    }
    



}


bool read_image(ifstream& in, std::array<std::uint8_t, 28*28>& img) {
    
    ifstream in("./data/cat", std::ios::binary);

    for (size_t i = 0; i < img.size(); ++i) {
        int ch = in.get();
        if (ch == EOF) return false;                 // finished (or truncated)
        img[i] = static_cast<std::uint8_t>(ch);
    }
    return true;
}

void test(){


    // // tests XOR

    // data_inp.resize(4,2);
    // data_t.resize(4,1);
    // data_inp << 0,0, 0,1, 1,0, 1,1;
    // data_t << 0, 1, 1, 0;

    // cout<<"before train"<<endl;
    // cout<<"Weights:\nW_IH\n"<<nn.W_IH<<"\nW_H\n"<<nn.W_HO<<endl; 
    // for (int i = 0; i < 1000000; i++)
    // {
    //     r_int = rand_int(0,3);
    //     nn.inputs = data_inp.row(r_int).transpose();
    //     nn.target = data_t.row(r_int).transpose();
    //     nn.train(nn.inputs, nn.target);
    // }
    
    // cout<<"after train"<<endl;
    // cout<<"Weights:\nW_IH\n"<<nn.W_IH<<"\nW_H\n"<<nn.W_HO<<endl; 

    // cout<<"INFERENCE:\n";
    // for (int i = 0; i < 4; i++)
    // {
    //     nn.inputs = data_inp.row(i).transpose();
    //     nn.target = data_t.row(i).transpose();
    //     cout<<nn.inputs<<"\n\n"<<nn.think(nn.inputs)<<"\n\n";
        
    // }
}