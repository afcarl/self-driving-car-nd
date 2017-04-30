#include "lidar.h"

// mesearment dimension
const int n_z = 2;

// Laser measurement noise standard deviation position1 in m
const double std_px = 0.15;

// Laser measurement noise standard deviation position2 in m
const double std_py = 0.15;

lidar::lidar()
{
	R= MatrixXd(n_z, n_z);
	R << std_px*std_px, 0,
		 0, std_py*std_py;
}

VectorXd lidar::hx(const VectorXd& x)
{
	// lidar measures px, py which is just state[0:1]
	return x.head(2);
}
