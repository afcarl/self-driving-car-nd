#ifndef TOOLS_H_
#define TOOLS_H_
#include <vector>
#include "Eigen/Dense"

using namespace Eigen;

namespace Tools
{
	VectorXd CalculateRMSE(const std::vector<VectorXd> &estimations, const std::vector<VectorXd> &ground_truth);
	MatrixXd GenerateSigmaPoints(const VectorXd& X, const MatrixXd& P, const MatrixXd& Q, double lambda);
};

#endif /* TOOLS_H_ */
