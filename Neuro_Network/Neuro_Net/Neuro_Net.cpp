#include "Neuro_Net.h"
#include <iostream>
#include<cassert>




Neuro_Net::Neuro_Net(std::vector<int> &topology)
{	// this loop fills the layers
	for (int layernum = 0; layernum < topology.size();layernum++) {
		Layers.push_back(layer());
		int numberOutput;
		if (layernum == topology.size()) {
			numberOutput = 0;
		}
		else {
			numberOutput = topology[layernum];
		}
		//this loop fills the layer
		for (int neuronNum = 0; neuronNum <= topology[layernum]; neuronNum++) {
			Layers.back().push_back(Neuron(numberOutput, neuronNum));
		}

		Layers.back().back().SetOuputVAlue(1.0);
	}

}

void Neuro_Net::feedForward(std::vector<double> &inputvals) {
	assert(inputvals.size() == Layers[0].size() -1);
	for (int i = 0; i < inputvals.size();i++) {
		Layers[0][i].SetOuputVAlue(inputvals[i]);
	}
	//heres goes the foward propagation
	for (int layernum = 1; layernum < Layers.size(); layernum++) {
		layer &prevlayer = Layers[layernum - 1];
		for (int n = 0; n < Layers[layernum].size() -1;n++) {
			Layers[layernum][n].feedForward(prevlayer);
		}
	}
}

void Neuro_Net::backProp(std::vector<double> &targetVals) {
	// calculate the Root Mean Square Error
	layer &outputLayer = Layers.back();
	net_error = 0.0;
	for (int n = 0; n < outputLayer.size();n++) {
		double delta = targetVals[n] - outputLayer[n].getOutputVal();
		net_error += delta * delta;
	}
	net_error /= outputLayer.size() - 1;
	net_error = sqrt(net_error);

	for (int n = 0; n < outputLayer.size() - 1; n++) {
		outputLayer[n].calcOutputGrediants(targetVals[n]);
	}

	for (int layerNum = Layers.size() - 2; layerNum > 0; layerNum--) {
		layer &hiddenLayer = Layers[layerNum];
		layer &nextLayer = Layers[layerNum + 1];

		for (int n = 0; n < hiddenLayer.size(); n++) {
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}

	for (int LayerNum = Layers.size() - 1; LayerNum > 0; LayerNum--) {
		layer &current_layer = Layers[LayerNum];
		layer &prevLayer = Layers[LayerNum -1];

		for (int n = 0; n < current_layer.size(); n++) {
			current_layer[n].updateInputWeights(prevLayer);
		}
	}

	// Average measurment

	net_recentAverageError = (net_recentAverageError * net_recentAverageSmoothingFactor + net_error) / (net_recentAverageSmoothingFactor + 1.0);
}

void Neuro_Net::getResults(std::vector<double> &resultVals) {
	resultVals.clear();
	for (int n = 0; n < Layers.back().size(); n++) {
		resultVals.push_back(Layers.back()[n].getOutputVal());
	}

}

Neuro_Net::~Neuro_Net()
{
}
