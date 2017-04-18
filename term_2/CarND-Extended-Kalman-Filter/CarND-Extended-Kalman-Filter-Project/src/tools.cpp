#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
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

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
	
	MatrixXd Hj(3,4);
	Hj.setZero(3,4);
	
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);
	
	float c1 = px * px + py * py;
	float c2 = sqrt(c1);
	float c3 = c1*c2;
	
	if(fabs(c1) < 0.0001){
		std::cout << "CalculateJacobian () - Error - Division by Zero" << std::endl;
		return Hj;
	}
	
	Hj << (px/c2), (py/c2), 0, 0,
		  -(py/c1), (px/c1), 0, 0,
		  py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;
	
	return Hj;
}
