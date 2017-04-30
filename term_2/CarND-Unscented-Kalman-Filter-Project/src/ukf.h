#ifndef UKF_H
#define UKF_H

#include "Eigen/Dense"
#include <functional>

using Eigen::MatrixXd;
using Eigen::VectorXd;

typedef std::function<VectorXd(const VectorXd&, const VectorXd&)> Residual_Func;
typedef std::function<VectorXd(const VectorXd&, double)> Fx_Func;

class UKF
{
public:
	UKF(){};
	UKF(int n_x, int n_aug, Residual_Func residual_x, Residual_Func residual_z, Fx_Func Fx);
	virtual ~UKF();
	
	void prediction(double delta_t);
	
	typedef std::function<VectorXd(const VectorXd&)> Hx_func;
	void update(VectorXd z, const MatrixXd& R, Hx_func Hx);
	
	///* State dimension
	int n_x_;
	
	///* Augmented state dimension
	int n_aug_;
	
	///* Sigma point spreading parameter
	double lambda_;

	///* state vector
	VectorXd x_;

	///* state covariance matrix
	MatrixXd P_;
	
	MatrixXd Q_;

	///* predicted sigma points matrix
	MatrixXd Xsig_pred_;
	
	Residual_Func residual_x_func_;
	Residual_Func residual_z_func_;
	
	Fx_Func Fx_;

	///* Weights of sigma points
	VectorXd weights_;

	// normalised innovation squared
	double nis_;
		
private:
	MatrixXd cross_variance(const VectorXd& x, const VectorXd& z_pred, const MatrixXd& sigmas_x, const MatrixXd& sigmas_z);
};

#endif /* UKF_H */
