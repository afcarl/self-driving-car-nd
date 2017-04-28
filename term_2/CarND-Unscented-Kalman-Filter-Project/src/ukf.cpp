#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF():
	use_laser_(true),
	use_radar_(true),
	std_a_(0.2), //was 30
	std_yawdd_(0.2), //was 30
	std_laspx_(0.15),
	std_laspy_(0.15),
	std_radr_(0.3),
	std_radphi_(0.03),
	std_radrd_(0.3),
	n_x_(5),
	n_aug_(7),
	lambda_(3 - n_aug_),
	NIS_radar_(0), //FIXME
	NIS_laser_(0) //FIXME
{
	x_ = VectorXd(5);
	x_.fill(0.0);
	P_ = MatrixXd::Identity(5, 5);
	weights_ = VectorXd(2*n_aug_+1);
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package)
{
}

MatrixXd UKF::GenerateSigmaPoints()
{
	MatrixXd Xsig_out = MatrixXd(n_aug_, 2 * n_aug_ + 1);
	
	//create augmented mean vector
	VectorXd x_aug = VectorXd(n_aug_);
	
	//create augmented state covariance
	MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
	
	//create augmented mean state
	x_aug.fill(0.0);
	x_aug.head(n_x_) = x_;
	
	//create augmented covariance matrix
	MatrixXd Q = MatrixXd(2, 2);
	Q << std_a_ * std_a_, 0,
		 0, std_yawdd_ * std_yawdd_;
	P_aug.topLeftCorner(n_x_,n_x_) = P_;
	P_aug.bottomRightCorner(2,2) = Q;
	
	//create square root matrix
	MatrixXd L = P_aug.llt().matrixL();
	
	//create augmented sigma points
	Xsig_out.col(0)  = x_aug;
	for (int i = 0; i< n_aug_; i++)
	{
		Xsig_out.col(i + 1)       = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
		Xsig_out.col(i + 1+ n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
	}
	
	return Xsig_out;
}


MatrixXd UKF::PredictSigmaPoints(const MatrixXd& Xsig_aug, const double delta_t)
{
	//create matrix with predicted sigma points as columns
	MatrixXd Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);
	
	//predict sigma points
	for (int i = 0; i< 2*n_aug_+1; i++)
	{
		//extract values for better readability
		const double p_x = Xsig_aug(0,i);
		const double p_y = Xsig_aug(1,i);
		const double v = Xsig_aug(2,i);
		const double yaw = Xsig_aug(3,i);
		const double yawd = Xsig_aug(4,i);
		const double nu_a = Xsig_aug(5,i);
		const double nu_yawdd = Xsig_aug(6,i);
		
		//predicted state values
		double px_p, py_p;
		
		//avoid division by zero
		if (fabs(yawd) > 0.001) {
			px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
			py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
		}
		else {
			px_p = p_x + v*delta_t*cos(yaw);
			py_p = p_y + v*delta_t*sin(yaw);
		}
		
		double v_p = v;
		double yaw_p = yaw + yawd*delta_t;
		double yawd_p = yawd;
		
		//add noise
		px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
		py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
		v_p = v_p + nu_a*delta_t;
		
		yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
		yawd_p = yawd_p + nu_yawdd*delta_t;
		
		//write predicted sigma point into right column
		Xsig_pred(0,i) = px_p;
		Xsig_pred(1,i) = py_p;
		Xsig_pred(2,i) = v_p;
		Xsig_pred(3,i) = yaw_p;
		Xsig_pred(4,i) = yawd_p;
	}
	return Xsig_pred;
}

void UKF::PredictMeanAndCovariance(const MatrixXd& Xsig_pred)
{
	// set weights
	weights_(0) = lambda_/(lambda_+n_aug_);
	for (int i=1; i<2*n_aug_+1; i++)
	{
		double weight = 0.5/(lambda_ + n_aug_);
		weights_(i) = weight;
	}
	
	//predicted state mean
	for (int i = 0; i < 2 * n_aug_ + 1; i++)
		x_ = x_ + weights_(i) * Xsig_pred.col(i);
	
	//predicted state covariance matrix
	for (int i = 0; i < 2 * n_aug_ + 1; i++)
	{
		// state difference
		VectorXd x_diff = Xsig_pred.col(i) - x_;
		//angle normalization
		while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
		while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
		
		P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
	}
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t)
{
	//create sigma point matrix
	MatrixXd Xsig_aug = GenerateSigmaPoints();
	std::cout << "Xsig_aug = " << std::endl << Xsig_aug << std::endl;
	
	//predict sigma point matrix
	MatrixXd Xsig_pred = PredictSigmaPoints(Xsig_aug, delta_t);
	std::cout << "Xsig_pred = " << std::endl << Xsig_pred << std::endl;
	
	PredictMeanAndCovariance(Xsig_pred);
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
}
