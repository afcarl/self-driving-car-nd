#include "ctrv.h"
#include <iostream>

double ctrv::normalise_angle(double angle)
{
	return atan2(sin(angle), cos(angle));
// could use the following as well
//	while (angle > M_PI) angle -= 2.*M_PI;
//	while (angle < -M_PI) angle += 2.*M_PI;
//	return angle;
}

VectorXd ctrv::residual_x(const VectorXd& x_sig, const VectorXd& x)
{
	VectorXd diff = x_sig - x;
	diff(3) = normalise_angle(diff(3));
	return diff;
}

VectorXd ctrv::residual_z(const VectorXd& z_sig, const VectorXd& z)
{
	VectorXd diff = z_sig - z;
	diff(1) = normalise_angle(diff(1));
	return diff;
}

VectorXd ctrv::Fx(const VectorXd& x, double dt)
{
	VectorXd Xsig_pred(5);

	const double p_x = x(0);
	const double p_y = x(1);
	const double v = x(2);
	const double yaw = x(3);
	const double yawd = x(4);
	const double nu_a = x(5);
	const double nu_yawdd = x(6);
	
	//predicted state values
	double px_p, py_p;
	
	//avoid division by zero
	if (fabs(yawd) > 0.001) {
		px_p = p_x + v/yawd * ( sin (yaw + yawd*dt) - sin(yaw));
		py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*dt) );
	}
	else {
		px_p = p_x + v*dt*cos(yaw);
		py_p = p_y + v*dt*sin(yaw);
	}
	
	double v_p = v;
	double yaw_p = yaw + yawd*dt;
	double yawd_p = yawd;
	
	//add noise
	px_p = px_p + 0.5*nu_a*dt*dt * cos(yaw);
	py_p = py_p + 0.5*nu_a*dt*dt * sin(yaw);
	v_p = v_p + nu_a*dt;
	
	yaw_p = yaw_p + 0.5*nu_yawdd*dt*dt;
	yawd_p = yawd_p + nu_yawdd*dt;
	
	//write predicted sigma point into right column
	Xsig_pred << px_p, py_p, v_p, yaw_p, yawd_p;
	return Xsig_pred;
}
