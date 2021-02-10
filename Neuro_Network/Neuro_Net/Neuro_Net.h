#pragma once

#include "../Neuron/Neuron.h"
#include <vector>
#include <cmath>

typedef std::vector<Neuron> layer;

class Neuro_Net
{
private:
	std::vector<layer> Layers;
	double net_error;
	double net_recentAverageError;
	double net_recentAverageSmoothingFactor;
public:
	Neuro_Net(std::vector<int> &topology);
	void feedForward(std::vector<double> &inputvals);
	void backProp(std::vector<double> &inputvals);
	void getResults(std::vector<double> &resultVals);
	~Neuro_Net();
};

