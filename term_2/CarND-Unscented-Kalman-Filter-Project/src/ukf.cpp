#include "ukf.h"
#include "tools.h"
#include <iostream>

UKF::UKF(int n_x, int n_aug, Residual_Func residual_x, Residual_Func residual_z, Fx_Func Fx):
	n_x_(n_x),
	n_aug_(n_aug),
	lambda_(3 - n_aug_),
	residual_x_func_(residual_x),
	residual_z_func_(residual_z),
	Fx_(Fx),
	nis_(0)
{
	x_ = VectorXd::Zero(n_x_);
	P_ = MatrixXd::Identity(n_x_, n_x_);
	Xsig_pred_ = MatrixXd::Zero(n_x_, 2 * n_aug_ + 1);

	weights_ = VectorXd(2 *  n_aug_ + 1);
	weights_(0) = lambda_/(lambda_+n_aug_);
	for (int i=1; i < 2 * n_aug_ + 1; i++)
		weights_(i) = 0.5 / (lambda_ + n_aug_);
}

static std::tuple<VectorXd, MatrixXd> unscented_transform(const MatrixXd& weights, const MatrixXd& sig_points, Residual_Func& residual_func)
{
	const int n_x = sig_points.rows();
	VectorXd x = VectorXd::Zero(n_x);
	for (int i = 0; i < sig_points.cols(); i++)
		x = x + weights(i) * sig_points.col(i);
	
	MatrixXd P = MatrixXd::Zero(n_x, n_x);
	for (int i = 0; i < sig_points.cols(); i++)
	{
		VectorXd x_diff = residual_func(sig_points.col(i), x);
		P = P + weights(i) * x_diff * x_diff.transpose();
	}
	return std::make_tuple(x, P);
}

void UKF::prediction(double delta_t)
{
	//create sigma point matrix
	MatrixXd Xsig_aug = Tools::GenerateSigmaPoints(x_, P_, Q_, lambda_);

	//predict sigma point matrix
	for (int i = 0; i < Xsig_aug.cols(); i++)
		Xsig_pred_.col(i) = Fx_(Xsig_aug.col(i), delta_t);
	
	std::tie(x_, P_) = unscented_transform(weights_, Xsig_pred_, residual_x_func_);
}

void UKF::update(VectorXd z, const MatrixXd& R, Hx_func Hx)
{
	//create matrix for sigma points in measurement space
	MatrixXd Zsig = MatrixXd(z.size(), 2 * n_aug_ + 1);
	for (int i = 0; i < 2 * n_aug_ + 1; i ++)
		Zsig.col(i) = Hx(Xsig_pred_.col(i));
	
	MatrixXd S;
	VectorXd z_pred;
	std::tie(z_pred, S) = unscented_transform(weights_, Zsig, residual_z_func_);
	S = S + R;

	MatrixXd Tc = cross_variance(x_, z_pred, Xsig_pred_, Zsig);

	//Kalman gain K
	MatrixXd K = Tc * S.inverse();
	
	//residual
	VectorXd z_diff = residual_z_func_(z, z_pred);
	
	//update state mean and covariance matrix
	x_ = x_ + K * z_diff;
	P_ = P_ - K * S * K.transpose();
	
	//calculate nis
	nis_ = (z - z_pred).transpose() * S.inverse() * (z - z_pred);
}

MatrixXd UKF::cross_variance(const VectorXd& x, const VectorXd& z_pred, const MatrixXd& sigmas_x, const MatrixXd& sigmas_z)
{
	//create matrix for cross correlation Tc
	MatrixXd Tc = MatrixXd::Zero(x.size(), z_pred.size());
	
	//calculate cross correlation matrix
	for (int i = 0; i < sigmas_x.cols(); i++)
	{
		VectorXd z_diff = residual_z_func_(sigmas_z.col(i), z_pred);
		VectorXd x_diff = residual_x_func_(sigmas_x.col(i), x);
		Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
	}
	return Tc;
}



