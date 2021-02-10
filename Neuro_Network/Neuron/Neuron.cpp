#include "Neuron.h"



Neuron::Neuron(int Numberofoutput, int Index)
{
	for (int i = 0; i < Numberofoutput; i++) {
		ouput_weigths.push_back(Conections());
		ouput_weigths.back().weight = randomWeight();
	}
	my_Index = Index;

}

void Neuron::feedForward(layer &prevLayer) {
	double sum = 0.0;
	for (int n = 0; n < prevLayer.size(); n++) {
		sum += prevLayer[n].getOutputVal() * prevLayer[n].ouput_weigths[my_Index].weight;
	}
	output_value = Neuron::transferFunction(sum);


}

double Neuron::transferFunction(double x) {
	// tanh - output range [-1,0..1,0]
	return tanh(x);
}

double Neuron::transferFunctionDerivative(double x) {
	// tanh derivative
	return 1 - x * x;
}

void Neuron::calcOutputGrediants(double targetVal) {
	double delta = targetVal - output_value;
	gradiant = delta * Neuron::transferFunctionDerivative(output_value);
}

void Neuron::calcHiddenGradients(layer &nextLayer) {
	double dow = sumDow(nextLayer);
	gradiant = dow * Neuron::transferFunctionDerivative(output_value);
}

double Neuron::sumDow(layer &nextLayer) {

	double sum = 0.0;

	for (unsigned n = 0; n < nextLayer.size() - 1; n++) {
		sum += ouput_weigths[n].weight;
	}
	return sum;
}

void Neuron::updateInputWeights(layer &prevLayer) {
	for (int n = 0; n < prevLayer.size(); n++) {
		Neuron &neuron = prevLayer[n];
		double oldDeltaWeight = neuron.ouput_weigths[my_Index].deltaweight;

		double newDeltaWeight = eta * neuron.getOutputVal() * gradiant + alpha * oldDeltaWeight;

		neuron.ouput_weigths[my_Index].deltaweight = newDeltaWeight;
		neuron.ouput_weigths[my_Index].weight += newDeltaWeight;
	}
}

Neuron::~Neuron()
{
}
