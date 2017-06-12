#ifndef MPC_H
#define MPC_H

#include <vector>
#include "Eigen-3.3/Eigen/Core"

class MPC
{
public:
  MPC();

  virtual ~MPC();

  // Solve the model given an initial state and polynomial coefficients
  // Return full solution data calculated by ipopt
  struct Solution
  {
	  using VectorD = std::vector<double>;
	  VectorD x;
	  VectorD y;
	  VectorD psi;
	  VectorD v;
	  VectorD cte;
	  VectorD epsi;
	  double delta;
	  double a;
  };

  Solution Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs);
};



#endif /* MPC_H */
