#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
	is_initialized_ = false;

	previous_timestamp_ = 0;

	// initializing matrices
	R_laser_ = MatrixXd(2, 2);
	R_radar_ = MatrixXd(3, 3);
	H_laser_ = MatrixXd(2, 4);
	Hj_ = MatrixXd(3, 4);
	
	//measurement covariance matrix - laser
	R_laser_ << 0.0225, 0,
			   0, 0.0225;

	//measurement covariance matrix - radar
	R_radar_ << 0.09, 0, 0,
			   0, 0.0009, 0,
			   0, 0, 0.09;
	
	// state covariance matrix P
	ekf_.P_ = MatrixXd(4, 4);
	ekf_.P_ << 1, 0, 0, 0,
			   0, 1, 0, 0,
			   0, 0, 1000, 0,
			   0, 0, 0, 1000;
	
	// state transition matrix
	ekf_.F_ = MatrixXd(4, 4);
	ekf_.F_ << 1, 0, 1, 0,
				0, 1, 0, 1,
				0, 0, 1, 0,
				0, 0, 0, 1;
}

FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {

	// Initilisation
	if (!is_initialized_){
		// first measurement
		cout << "EKF: " << endl;
		ekf_.x_ = VectorXd(4);

		if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
			float ro = measurement_pack.raw_measurements_[0];
			float phi = measurement_pack.raw_measurements_[1];
			ekf_.x_ << ro*cos(phi), ro*sin(phi), 0, 0;
		}
		else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
			ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
		}
			
		previous_timestamp_ = measurement_pack.timestamp_;

		// done initializing, no need to predict or update
		is_initialized_ = true;
		return;
	}

	// Prediction
	
	//compute the time elapsed between the current and previous measurements
	float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
	previous_timestamp_ = measurement_pack.timestamp_;

	float dt_2 = dt * dt;
	float dt_3 = dt_2 * dt;
	float dt_4 = dt_3 * dt;

	//Modify the F matrix so that the time is integrated
	
	ekf_.F_(0, 2) = dt;
	ekf_.F_(1, 3) = dt;
	
	float noise_ax = 9;
	float noise_ay = 9;
	
	//set the process covariance matrix Q
	MatrixXd Q = MatrixXd(4, 4);
	Q <<  dt_4/4*noise_ax, 0, dt_3/2*noise_ax, 0,
		  0, dt_4/4*noise_ay, 0, dt_3/2*noise_ay,
		  dt_3/2*noise_ax, 0, dt_2*noise_ax, 0,
		  0, dt_3/2*noise_ay, 0, dt_2*noise_ay;
	ekf_.Q_ = Q;
	ekf_.Predict();

	// Update
	
	if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
		// Radar updates
		ekf_.R_ = R_radar_;
		Hj_ = tools.CalculateJacobian(ekf_.x_);
		if (!Hj_.isZero(0))
		{
			ekf_.H_ = Hj_;
			ekf_.UpdateEKF(measurement_pack.raw_measurements_);
		}
	} else {
		// Laser updates
		ekf_.R_ = R_laser_;
		H_laser_ << 1, 0, 0, 0,
					 0, 1, 0, 0;
		ekf_.H_ = H_laser_;
		ekf_.Update(measurement_pack.raw_measurements_);
	}
	// print the output
	cout << "x_ = " << ekf_.x_ << endl;
	cout << "P_ = " << ekf_.P_ << endl;
}
