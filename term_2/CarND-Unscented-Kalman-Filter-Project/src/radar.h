#pragma once

#include "Eigen/Dense"

using Eigen::VectorXd;
using Eigen::MatrixXd;

struct radar
{
	radar();
	VectorXd hx(const VectorXd& x);
	MatrixXd R;
};