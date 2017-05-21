/*
 * particle_filter.cpp
 *
 *  Created on: May 18, 2017
 *      Author: Peter Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <map>

#include "particle_filter.h"

using namespace std;

static double multi_variate_gussian(double x, double mean_x, double delta_x, double y, double mean_y, double delta_y)
{
	return (1 / (2 * M_PI * delta_x * delta_y)) * exp(-(pow((x-mean_x),2) / (2*delta_x*delta_x) + pow((y-mean_y),2) / (2*delta_y*delta_y)));
}

void ParticleFilter::init(double x, double y, double theta, double std[])
{
	num_particles = 10;
	weights = std::vector<double>(num_particles, 1);
	
	const double std_x = std[0];
	const double std_y = std[1];
	const double std_theta = std[2];
	
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);
	
	default_random_engine gen;
	for (int i = 0; i < num_particles; i++)
	{
		Particle particle;
		particle.id = i;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
		particle.weight = 1;
		particles.push_back(particle);
	}
	
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
	std::random_device r;
	default_random_engine gen(r());
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

	for (auto& particle : particles)
	{
		double theta = particle.theta;

		if (abs(yaw_rate) > 0.001)
		{
			particle.x += (velocity / yaw_rate) * (sin(theta + yaw_rate * delta_t) - sin(theta)) + dist_x(gen);
			particle.y += (velocity / yaw_rate) * (cos(theta) - cos(theta + yaw_rate * delta_t)) + dist_y(gen);
			particle.theta += yaw_rate * delta_t + dist_theta(gen);
			//particle.theta = fmod(particle.theta, (2.*M_PI));
		}
		else
		{
			particle.x += velocity * delta_t * cos(theta)+ dist_x(gen);
			particle.y += velocity * delta_t * sin(theta) + dist_y(gen);
			particle.theta += dist_theta(gen);
		}
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations)
{
    // not used, this interface is somewhat confusing
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
									std::vector<LandmarkObs> observations, Map map_landmarks)
{
	weights.clear();

	for (auto& particle : particles)
	{
		// observation corodinate transoformation
		std::vector<LandmarkObs> observation_transformed;
		for (auto observation : observations)
		{
			//std::cout << observation.x << "," << observation.y << "," << observation.id << std::endl;
			double x_t = particle.x + observation.x * cos(particle.theta) - observation.y * sin(particle.theta);
			double y_t = particle.y + observation.x * sin(particle.theta) + observation.y * cos(particle.theta);
			observation_transformed.push_back({0, x_t, y_t});
		}
		
		// data association
		struct association
		{
			LandmarkObs observation;
			Map::single_landmark_s landmark;
		};
		std::vector<association> associations;
		for (auto observation : observation_transformed)
		{
			double smallest_distance = dist(observation.x, observation.y, map_landmarks.landmark_list[0].x_f, map_landmarks.landmark_list[0].y_f);
			Map::single_landmark_s bestMatch = map_landmarks.landmark_list[0];
			
			for (auto landmark : map_landmarks.landmark_list)
			{
				double distance = dist(observation.x, observation.y, landmark.x_f, landmark.y_f);
				if (distance < smallest_distance)
				{
					smallest_distance = distance;
					bestMatch = landmark;
				}
			}
			// FUDGE: if there are multiple observations for the same landmark need to sort
			associations.push_back({ observation, bestMatch});
		}
		
		// calculating weights
		double weight = 1;
		for (auto association : associations)
		{
			const double this_weight = multi_variate_gussian(association.observation.x, association.landmark.x_f, std_landmark[0],
										   		association.observation.y, association.landmark.y_f, std_landmark[1]);
			weight *= this_weight;
			
		}
		
		particle.weight = weight;
		weights.push_back(weight);
	}
}

void ParticleFilter::resample()
{
	std::random_device r;
	default_random_engine gen(r());
	std::discrete_distribution<> dist(weights.begin(), weights.end());
	std::vector<Particle> new_particles;

	for(int i = 0; i < num_particles; i++)
	{
		new_particles.push_back(particles[dist(gen)]);
	}
	
	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
