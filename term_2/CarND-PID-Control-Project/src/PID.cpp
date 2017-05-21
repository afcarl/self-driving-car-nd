#include "PID.h"
#include <algorithm>

using namespace std;

template <typename T>
T bound(T val, T low, T high) noexcept
{
    val = std::min(val, high);
    val = std::max(val, low);
    return val;
}

PID::PID(): 
   p_error(0),
   i_error(0),
   d_error(0),
   Kp(0),
   Ki(0),
   Kd(0) 
{
}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd)
{
    Kp = Kp;
    Ki = Ki;
    Kd = Kd;
}

void PID::UpdateError(double cte) 
{
    const int dt = 0.1; //100 miliseconds time step
    p_error = Kp * cte;
    i_error += cte * dt;
    i_error = bound(i_error, -1.0, 1.0);
    d_error = cte / dt - d_error;   
}

double PID::TotalError()
{
    double totalError = Kp*p_error + Ki*i_error + Kd*d_error;  
    totalError = bound(totalError, -1.0, 1.0);
    return totalError;
}

