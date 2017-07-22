#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include "classifier.h"

/**
 * Initializes GNB
 */
GNB::GNB() {

}

GNB::~GNB() {}


static double gussian(double x, double mean_x, double delta_x)
{
    return (1 / (2 * M_PI * delta_x)) * exp(-(pow((x-mean_x),2) / (2*delta_x*delta_x)));
}

void GNB::train(vector<vector<double>> data, vector<string> labels)
{ 
    map<string, vector<double>> s_distributionMap;
    map<string, vector<double>> d_distributionMap;
    map<string, vector<double>> sdot_distributionMap;
    map<string, vector<double>> ddot_distributionMap;
    
    assert(data.size() == labels.size());
    
    for (size_t i = 0; i < labels.size(); i++)
    {
        string lable = lables[i];
        s_distrubitionMap[lable].push_back(data[i][0]);
        d_distributionMap[label].push_back(data[i][1]);
        sdot_distrubitionMap[lable].push_back(data[i][2]);
        ddot_distributionMap[label].push_back(data[i][3]);
    }
    
    m_distributionsMap.clear();
    // {left : [s_gussian, d_gussian, sdot_gussian, ddot_gussian]}
    for (auto label : possible_labels)
    {
        m_distributionsMap[label].push_back(calMeanandVariance(s_distributionMap[label]));
        m_distributionsMap[label].push_back(calMeanandVariance(d_distributionMap[label]));
        m_distributionsMap[label].push_back(calMeanandVariance(sdot_distributionMap[label]));
        m_distributionsMap[label].push_back(calMeanandVariance(ddot_distributionMap[label]));
    }
}

double GNB::getProbabilityForLabel(const string& label, const vector<double>& data)
{
    double probability = 1;
    auto distributions = m_distributionsMap[label];
    for (size_t i = 0; i < distributions.size(); i++)
    {
        probability *= gussian(data[i], distributions[i].mean, distributions[i].variance);
    }
    return probability;
}

string GNB::predict(vector<double> testData)
{
    int best_index = 0;
    double highest_p = 0; 

    for (i = 0; i < possible_lables.size(); i++)
    {
        string lable = possible_lables[i];
        double p = getProbabilityForLabel(lable, testData);
        if (p > highest_p)
        {
            highest_p = p;
            best_index = i;
        }
    }
    
    return possible_labels[best_index];
}
