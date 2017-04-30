#include "fushion.h"
#include "ctrv.h"

Fushion::Fushion():
	is_initialized_(false),
	use_laser_(true),
	use_radar_(true),
	previous_timestamp_(0)
{
	using namespace std::placeholders;
	ukf_ = UKF(5, 7, std::bind(ctrv::residual_x, _1, _2),
					 std::bind(ctrv::residual_z, _1, _2),
			         std::bind(ctrv::Fx, _1, _2));
	
	// Process noise standard deviation longitudinal acceleration in m/s^2
	const double std_a = 0.2; //was 30
	
	// Process noise standard deviation yaw acceleration in rad/s^2
	const double std_yawdd = 0.2; //was 30
	
	// process noise
	ukf_.Q_ = MatrixXd(2, 2);
	ukf_.Q_ << std_a * std_a, 0,
			   0, std_yawdd * std_yawdd;
}

void Fushion::ProcessMeasurement(MeasurementPackage measurement)
{
	//Initilisation
	if (!is_initialized_)
	{
		if (measurement.sensor_type_ == MeasurementPackage::RADAR)
		{
			float ro = measurement.raw_measurements_[0];
			float phi = measurement.raw_measurements_[1];
			ukf_.x_ << ro*cos(phi), ro*sin(phi), 0, 0, 0;
		}
		else if (measurement.sensor_type_ == MeasurementPackage::LASER)
		{
			ukf_.x_ << measurement.raw_measurements_[0], measurement.raw_measurements_[1], 0, 0;
		}
		
		previous_timestamp_ = measurement.timestamp_;
		is_initialized_ = true;
		return;
	}
	
	//Compute the time elapsed between the current and previous measurements
	float dt = (measurement.timestamp_ - previous_timestamp_) / 1000000.0; //in seconds
	previous_timestamp_ = measurement.timestamp_;
	
	//Prediction
	ukf_.prediction(dt);
	
	//Update
	if ((measurement.sensor_type_ == MeasurementPackage::RADAR) && use_radar_)
	{
		ukf_.update(measurement.raw_measurements_, radar_.R, std::bind(&radar::hx, radar_, std::placeholders::_1));
		nis_radar_ = ukf_.nis_;
	}
	else if (measurement.sensor_type_ == MeasurementPackage::LASER && use_laser_)
	{
		ukf_.update(measurement.raw_measurements_, lidar_.R, std::bind(&lidar::hx, lidar_, std::placeholders::_1));
		nis_lidar_ = ukf_.nis_;
	}
}

