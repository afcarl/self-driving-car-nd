#pragma once

#include <algorithm>
/*
   Effect of each component
   1. P term describe how much the system response to the error propotianally.
   2. I term can increase the settle time, but it coul dovershoot if toolarge. It is mainly used to eliminate steady state error. i.e system can not reach setpoint for some reasons (vehicle wheel not calibrated well, a drone need more accumulated thrust to reach hover setpoint as battery power is decreasing. etc.
   3. D term is prediction. It is the error's change rate which conteracts with the effect of P and I. it can effectively reduce oscillation,      It is like a spring or an inductor which opposes the rate of change of current flowing through it. That is why it can achieve smooth steering control. However, too large cause overdamped system, slower settle time, and cause system to chatter.
*/

/* The following steps are used to tune PID:
1. Determine control frequency which is used to calculate time step dt = 1/frequency. Control frequency depends on system's dynamics, this is usually determined empirically. In this simulator, it is measured to be 80Hz.
2. Set all gains to zero to start with.
3. Increase the P gain until the car starts oscillates around the expected path (center of the road).
4. Increase the D gain until the the oscillations goes away (i.e. it's critically damped).
5. Repeat steps 2 and 3 until increasing the D gain does not stop the oscillations.
6. Set P and D to the last stable values.
7. Increase the I gain until it brings you the the set point if there is steady state error. In simulation there isn't, but it can be used to increase response time to settle.) PD is enough to have a good drive on the entire track already. 
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
