#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth)
{
	assert((estimations.size() == ground_truth.size()) || estimations.size() != 0);
	
	VectorXd rmse(4);
	rmse << 0,0,0,0;
	
	for(int i=0; i < estimations.size(); ++i){
		VectorXd error = estimations[i] - ground_truth[i];
		error = error.array() * error.array();
		rmse += error;
	}
	
	rmse = rmse.array() / estimations.size();
	rmse = rmse.array().sqrt();
	
	return rmse;
}
