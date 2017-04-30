#pragma once

#include "ukf.h"
#include "radar.h"
#include "lidar.h"
#include "measurement_package.h"

class Fushion
{
public:
	Fushion();
	void ProcessMeasurement(MeasurementPackage meas_package);
	
	UKF ukf_;
	double nis_radar_;
	double nis_lidar_;

private:
	radar radar_;
	lidar lidar_;
	bool is_initialized_;
	bool use_laser_;
	bool use_radar_;
	long previous_timestamp_;
};