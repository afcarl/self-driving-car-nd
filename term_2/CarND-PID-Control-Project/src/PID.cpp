#include "PID.h"

PID::PID(PIDParameters parameter):
	m_parameter(parameter),
	m_sum(0),
	m_previousError(0)
{
}

double PID::compute(double error)
{
	m_sum += error * m_parameter.dt;
	m_sum = bound<double>(m_sum, m_parameter.integralLowLimit, m_parameter.integralHighLimit); // prevent integral windup
	const double deriv = (error - m_previousError) / m_parameter.dt;
	m_previousError = error;
	return m_parameter.Kp * error + m_parameter.Ki * m_sum + m_parameter.Kd * deriv;
}
