#pragma once

#include "Eigen/Dense"

using Eigen::VectorXd;
using Eigen::MatrixXd;

struct lidar
{
	lidar();
	VectorXd hx(const VectorXd& x);
	MatrixXd R;
};