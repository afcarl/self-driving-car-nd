#pragma once

#include <algorithm>

/* To tune a PID use the following steps:
1. Set all gains to zero.
2. Increase the P gain until the response to a disturbance is steady oscillation.
3. Increase the D gain until the the oscillations go away (i.e. it's critically damped).
4. Repeat steps 2 and 3 until increasing the D gain does not stop the oscillations.
5. Set P and D to the last stable values.
6. Increase the I gain until it brings you to the setpoint with the number of oscillations desired (normally zero but a quicker response can be had if you don't mind a couple oscillations of overshoot)
 
What disturbance you use depends on the mechanism the controller is attached to. Normally moving the mechanism by hand away from the setpoint and letting go is enough. If the oscillations grow bigger and bigger then you need to reduce the P gain.
																																						 
If you set the D gain too high the system will begin to chatter (vibrate at a higher frequency than the P gain oscillations). If this happens, reduce the D gain until stops.
*/

template <typename T>
T bound(T val, T low, T high) noexcept
{
	val = std::min(val, high);
	val = std::max(val, low);
	return val;
}

struct PIDParameters
{
	double Kp;
	double Ki;
	double Kd;
	double dt;
	double integralLowLimit;
	double integralHighLimit;
};

class PID
{
public:
	PID(PIDParameters p);
	double compute(double error);
	
private:
	PIDParameters m_parameter;
	double m_sum;
	double m_previousError;
};
