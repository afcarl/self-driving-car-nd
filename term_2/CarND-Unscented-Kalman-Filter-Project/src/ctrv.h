#pragma once

#include "Eigen/Dense"

using Eigen::VectorXd;

namespace ctrv
{
	void normalise_angle(double& angle);
	VectorXd residual_x(const VectorXd& x_sig, const VectorXd& x);
	VectorXd residual_z(const VectorXd& z_sig, const VectorXd& z);
	//state vector x [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
	VectorXd Fx(const VectorXd& x, double dt);
};