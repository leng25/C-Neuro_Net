#pragma once
#include<vector>
#include<cstdlib>
#include <cmath>




struct Conections {
	double weight;
	double deltaweight;
};

class Neuron
{
typedef std::vector<Neuron> layer;

private:
	double eta = 0.15;
	double alpha = 0.5;
	static double randomWeight(void) { return rand() / double(RAND_MAX); }
	double output_value;
	std::vector<Conections> ouput_weigths;
	int my_Index;
	double gradiant;
	static double transferFunction(double x);
	static double transferFunctionDerivative(double x);
	double sumDow(layer &nextLayer);
public:
	Neuron(int Numberofoutput, int my_Index);
	void feedForward(layer &prevLayer);
	void SetOuputVAlue(double val) { output_value = val; }
	double getOutputVal(void) { return output_value; }
	void calcOutputGrediants(double targetvalue);
	void calcHiddenGradients(layer &nextLayer);
	void updateInputWeights(layer &prevLayer);
	~Neuron();
};

