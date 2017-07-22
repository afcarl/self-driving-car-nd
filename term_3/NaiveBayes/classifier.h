#ifndef CLASSIFIER_H
#define CLASSIFIER_H
#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include <map>
#include <cstdint>

using namespace std;

class GNB {
public:

	map<string> possible_labels = {"left","keep","right"};

   	/**
  	* Constructor
  	*/
 	GNB();

	/**
 	* Destructor
 	*/
 	virtual ~GNB();

 	void train(vector<vector<double> > data, vector<string>  labels);

  	string predict(vector<double> testData);

private:
    double getProbabilityForLabel(const string& label, const vector<double>& data); 
    
    enum class feature: uint8_t 
    {
        ks,
        kd,
        ks_dot,
        kd_dot,
    };

    struct Gussian
    {
        double mean;
        double variance;
    };
    
    Gussian m_featue_disribution[3]; 
    map<feature,  > m_featureDistributionMap;

    map<>
};

#endif




