#include "../Neuro_Network/Neuro_Net/Neuro_Net.h"
#include <vector>

int main() {
	std::vector<int> topology;
	topology.push_back(3);
	topology.push_back(2);
	topology.push_back(1);
	Neuro_Net myNet(topology);
	return 0;
}