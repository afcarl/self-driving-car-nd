#include "radar.h"

// mesearment dimension
const int n_z = 3;

///* Radar measurement noise standard deviation radius in m
const double std_r = 0.3;

///* Radar measurement noise standard deviation angle in rad
const double std_phi = 0.03;

///* Radar measurement noise standard deviation radius change in m/s
const double std_rdot = 0.3;

radar::radar()
{
	R = MatrixXd(n_z, n_z);
	R << std_r*std_r, 0, 0,
		 0, std_phi*std_phi, 0,
		 0, 0, std_rdot*std_rdot;
}

VectorXd radar::hx(const VectorXd& x)
{
	VectorXd hx(3);
	const float px = x[0];
	const float py = x[1];
	const float v = x[2];
	const float yaw = x[3];
	const float rho = sqrt(px*px + py*py);
	const float phi = atan2(py, px);
	float rho_dot;
	if (rho > 0.001)
	{
	 	rho_dot = (px*cos(yaw)*v + py*sin(yaw)*v) / rho;
	}
	else
		rho_dot = 0.0;
	hx << rho, phi, rho_dot;
	return hx;
}
