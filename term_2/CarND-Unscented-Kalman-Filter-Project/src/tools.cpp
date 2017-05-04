#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;

VectorXd Tools::CalculateRMSE(const std::vector<VectorXd> &estimations,
                              const std::vector<VectorXd> &ground_truth)
{
	assert((estimations.size() == ground_truth.size()) || estimations.size() != 0);
	
	VectorXd rmse = VectorXd::Zero(4);
	
	for(int i = 0; i < estimations.size(); ++i)
	{
		VectorXd error = estimations[i] - ground_truth[i];
		error = error.array() * error.array();
		rmse += error;
	}
	
	rmse = rmse.array() / estimations.size();
	rmse = rmse.array().sqrt();
	
	return rmse;
}

MatrixXd Tools::GenerateSigmaPoints(const VectorXd& X, const MatrixXd& P, const MatrixXd& Q, double lambda)
{
	const int n_aug = P.rows() + Q.rows();
	const int n_sig = 2 * n_aug + 1;
	const int n_x = X.size();
	MatrixXd Xsig_out = MatrixXd(n_aug, n_sig);
	
	//create augmented mean vector
	VectorXd x_aug = VectorXd::Zero(n_aug);
	x_aug.head(n_x) = X;
	
	//create augmented state covariance
	MatrixXd P_aug = MatrixXd::Zero(n_aug, n_aug);
	
	//create augmented covariance matrix
	P_aug.topLeftCorner(n_x, n_x) = P;
	P_aug.bottomRightCorner(2,2) = Q;
	
	//create square root matrix
	MatrixXd L = P_aug.llt().matrixL();
	
	//create augmented sigma points
	Xsig_out.col(0)  = x_aug;
	for (int i = 0; i < n_aug; i++)
	{
		Xsig_out.col(i + 1)       = x_aug + sqrt(lambda + n_aug) * L.col(i);
		Xsig_out.col(i + 1+ n_aug) = x_aug - sqrt(lambda + n_aug) * L.col(i);
	}
	return Xsig_out;
}

//MatrixXd Tools::GenerateWeights()
