#include <fstream>
#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "Eigen-3.3/Eigen/LU"
#include "json.hpp"
#include "spline.h"
#include "matplotlibcpp.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
	auto found_null = s.find("null");
	auto b1 = s.find_first_of("[");
	auto b2 = s.find_first_of("}");
	if (found_null != string::npos) {
		return "";
	} else if (b1 != string::npos && b2 != string::npos) {
		return s.substr(b1, b2 - b1 + 2);
	}
	return "";
}

const int kLanewidth = 4;
const int kControlFrequency = 50;
const double kControlInterval = 1.0 / kControlFrequency;
const double kTargetSpeedMS = 20.5; // m/s approcimately 48mph
const int kMaxAccelerationTimeToTarget = 4; // seconds
const int kMaxChangeLaneTime = 4; // seconds
const double kMinSpeedError = 0.01; // m/s
const double kMinLaneError = 0.00001; // d in meter
const double kMinSafeDistance = 15; // meter

struct Car
{
	double x;
	double y;
	double s;
	double d;
	double speed;
	double theta;
	int current_lane;
	// each vehicle has [ id, x, y, vx, vy, s, d]
	vector<vector<double>> sensor;
};

struct MapInfo
{
	vector<double> waypoints_x;
	vector<double> waypoints_y;
	vector<double> waypoints_s;
	vector<double> waypoints_dx;
	vector<double> waypoints_dy;
};

vector<double> JMT(vector< double> start, vector <double> end, double T)
{
	/*
	 Calculate the Jerk Minimizing Trajectory that connects the initial state
	 to the final state in time T.
	 start/end - the vehicles start location given as a length three array
	 corresponding to initial values of [s, s_dot, s_double_dot]
	 */
	
	MatrixXd A = MatrixXd(3, 3);
	A << T*T*T, T*T*T*T, T*T*T*T*T,
	3*T*T, 4*T*T*T,5*T*T*T*T,
	6*T, 12*T*T, 20*T*T*T;
	
	MatrixXd B = MatrixXd(3,1);
	B << end[0]-(start[0]+start[1]*T+.5*start[2]*T*T),
	end[1]-(start[1]+start[2]*T),
	end[2]-start[2];
	
	MatrixXd Ai = A.inverse();
	
	MatrixXd C = Ai*B;
	
	vector <double> result = {start[0], start[1], .5*start[2]};
	for(int i = 0; i < C.size(); i++)
		result.push_back(C.data()[i]);
	
	return result;
}

double distance(double x1, double y1, double x2, double y2)
{
	return sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
}

int ClosestWaypoint(double x, double y, vector<double> maps_x, vector<double> maps_y)
{
	double closestLen = 100000; //large number
	int closestWaypoint = 0;
	
	for(int i = 0; i < maps_x.size(); i++)
	{
		double map_x = maps_x[i];
		double map_y = maps_y[i];
		double dist = distance(x,y,map_x,map_y);
		if(dist < closestLen)
		{
			closestLen = dist;
			closestWaypoint = i;
		}
	}
	
	return closestWaypoint;
}

int NextWaypoint(double x, double y, double theta, vector<double> maps_x, vector<double> maps_y)
{
	int closestWaypoint = ClosestWaypoint(x,y,maps_x,maps_y);
	
	double map_x = maps_x[closestWaypoint];
	double map_y = maps_y[closestWaypoint];
	
	double heading = atan2( (map_y-y),(map_x-x) );
	
	double angle = abs(theta-heading);
	
	if(angle > pi()/4)
	{
		closestWaypoint++;
	}
	
	return closestWaypoint;
}

// Transform from Cartesian x,y coordinates to Frenet s,d coordinates
vector<double> getFrenet(double x, double y, double theta, vector<double> maps_x, vector<double> maps_y)
{
	int next_wp = NextWaypoint(x,y, theta, maps_x,maps_y);
	
	int prev_wp;
	prev_wp = next_wp-1;
	if(next_wp == 0)
	{
		prev_wp  = maps_x.size()-1;
	}
	
	double n_x = maps_x[next_wp]-maps_x[prev_wp];
	double n_y = maps_y[next_wp]-maps_y[prev_wp];
	double x_x = x - maps_x[prev_wp];
	double x_y = y - maps_y[prev_wp];
	
	// find the projection of x onto n
	double proj_norm = (x_x*n_x+x_y*n_y)/(n_x*n_x+n_y*n_y);
	double proj_x = proj_norm*n_x;
	double proj_y = proj_norm*n_y;
	
	double frenet_d = distance(x_x,x_y,proj_x,proj_y);
	
	//see if d value is positive or negative by comparing it to a center point
	
	double center_x = 1000-maps_x[prev_wp];
	double center_y = 2000-maps_y[prev_wp];
	double centerToPos = distance(center_x,center_y,x_x,x_y);
	double centerToRef = distance(center_x,center_y,proj_x,proj_y);
	
	if(centerToPos <= centerToRef)
	{
		frenet_d *= -1;
	}
	
	// calculate s value
	double frenet_s = 0;
	for(int i = 0; i < prev_wp; i++)
	{
		frenet_s += distance(maps_x[i],maps_y[i],maps_x[i+1],maps_y[i+1]);
	}
	
	frenet_s += distance(0,0,proj_x,proj_y);
	
	return {frenet_s,frenet_d};
}

// Transform from Frenet s,d coordinates to Cartesian x,y
vector<double> getXY(double s, double d, vector<double> maps_s, vector<double> maps_x, vector<double> maps_y)
{
	int prev_wp = -1;
	
	while(s > maps_s[prev_wp+1] && (prev_wp < (int)(maps_s.size()-1) ))
	{
		prev_wp++;
	}
	
	int wp2 = (prev_wp+1)%maps_x.size();
	
	double heading = atan2((maps_y[wp2]-maps_y[prev_wp]),(maps_x[wp2]-maps_x[prev_wp]));
	// the x,y,s along the segment
	double seg_s = (s-maps_s[prev_wp]);
	
	double seg_x = maps_x[prev_wp]+seg_s*cos(heading);
	double seg_y = maps_y[prev_wp]+seg_s*sin(heading);
	
	double perp_heading = heading-pi()/2;
	
	double x = seg_x + d*cos(perp_heading);
	double y = seg_y + d*sin(perp_heading);
	
	return {x,y};
}

vector<double> worldToBody(double x, double y, double theta, double px, double py)
{
	double x_b = px - x;
	double y_b = py - y;
	double px_b = x_b * cos(theta) + y_b * sin(theta);
	double py_b = -x_b * sin(theta) + y_b * cos(theta);
	return {px_b, py_b};
}

vector<double> bodyToWorld(double x, double y, double theta, double px_b, double py_b)
{
	double px = x + px_b * cos(theta) - py_b * sin(theta);
	double py = y + px_b * sin(theta) + py_b * cos(theta);
	return {px, py};
}

double changeLane(int time, int current_lane, bool right)
{
	int k = right ? 1 : -1;
	return current_lane + k / (1 + exp(-0.03 * (time - 190))); // logistic function
}

vector<int> findNextWayPointsIndexList(const MapInfo& mapInfo, const Car& car)
{
	const int kMaxNextWayPoints = 10;
	vector<int> wayPointsIndex;
	vector<double> wayPoints_x, wayPoints_y;
	int nextWayPoint = ClosestWaypoint(car.x, car.y, mapInfo.waypoints_x, mapInfo.waypoints_y); // the first waypoint index of a waypoints segment
	nextWayPoint = nextWayPoint - 4;
	
	if (nextWayPoint < 0)
		nextWayPoint = mapInfo.waypoints_x.size() + nextWayPoint;
	
	for (int i = 0, nextwayPointIndex = nextWayPoint; i < kMaxNextWayPoints; i++)
	{
		wayPointsIndex.push_back(nextwayPointIndex);
		wayPoints_x.push_back(mapInfo.waypoints_x[nextwayPointIndex]);
		wayPoints_y.push_back(mapInfo.waypoints_y[nextwayPointIndex]);
		nextwayPointIndex++;
		nextwayPointIndex = nextwayPointIndex % (mapInfo.waypoints_x.size());
	}
	
	return wayPointsIndex;
}

void getSplineForWayPoints(const MapInfo& mapInfo, const vector<int>& wayPointsIndex, tk::spline& xb_s, tk::spline& yb_s, double next_d)
{
	static int cycle = 0;
	static int previous_last_index = -1;
	// The max s value before wrapping around the track back to 0
	const double kMax_s = 6945.554;
	
	// detecting wrap around
	if (previous_last_index != wayPointsIndex.back())
	{
		previous_last_index = wayPointsIndex.back();
		if (wayPointsIndex.back() == 0)
			cycle ++;
	}
	
	vector<double> lane_waypoints_x, lane_waypoints_y, 	lane_waypoints_s;
	for(auto index : wayPointsIndex)
	{
		double x = mapInfo.waypoints_x[index] + next_d * mapInfo.waypoints_dx[index];
		double y = mapInfo.waypoints_y[index] + next_d * mapInfo.waypoints_dy[index];
		lane_waypoints_x.push_back(x);
		lane_waypoints_y.push_back(y);
		// FIXME map lane's waypoints to frenet, using dx, dy
//		double theta = atan2(mapInfo.waypoints_dx[index],mapInfo.waypoints_dy[index]);
//		auto frenet = getFrenet(x, y, theta, mapInfo.waypoints_x, mapInfo.waypoints_y);
//		double s = frenet[0] + cycle * kMax_s;
		double s = mapInfo.waypoints_s[index] + cycle * kMax_s;
		lane_waypoints_s.push_back(s);
	}
	
	// correcting s value for previous cycle points
	if (cycle > 0)
	{
		int zero_index = std::find(wayPointsIndex.begin(), wayPointsIndex.end(), 0) - wayPointsIndex.begin();
		if (zero_index != wayPointsIndex.size()) //found
		{
			for (int i = 0; i < zero_index; i++)
			{
				lane_waypoints_s[i] -= kMax_s;
			}
		}
	}
	
	// build spline for interpolation
	xb_s.set_points(lane_waypoints_s, lane_waypoints_x);
	yb_s.set_points(lane_waypoints_s, lane_waypoints_y);
}

class Planner
{
public:
	Planner() {};
	
	void init(double car_s) {m_current_s = car_s;}
	
	void updateState(const Car& car)
	{
		vector<string> states = {"KL", "LCL", "LCR"};
		
		if (car.current_lane == 1)
			states.erase(states.begin() + 1);
		else if (car.current_lane == 3)
			states.erase(states.begin() + 2);
		
		string best_state;
		double lowest_cost = 999999;

		for (const auto state : states)
		{
			double cost = 0;
			int target_lane = getLaneForState(state, car);
			vector<double> front_vehicle = getVehicle(car, target_lane, true);
			vector<double> back_vehicle = getVehicle(car, target_lane, false);

			double front_vehicle_s, back_vehicle_s;
			front_vehicle_s = !front_vehicle.empty() ? front_vehicle[5] : 999999;
			back_vehicle_s = !back_vehicle.empty() && (target_lane != car.current_lane)? back_vehicle[5] : -999999;
			
			cost += 3 * pow(target_lane - car.current_lane, 2); //penalize lane changing
			cost += 25 * exp(-((front_vehicle_s - car.s) - 15) / 10); //reward lane change if target lane front part is clear
			cost += 30 * exp(-((car.s - back_vehicle_s) - 15) / 20); //reward lane change if target lane back part is clear
			
			// check collision and vehicle behind on other lanes
			double speed_diff = !back_vehicle.empty() ? vehicleSpeed(back_vehicle) - car.speed : -1;
			double collision_cost = speed_diff > 0 && (target_lane != car.current_lane) ? 100 * pow(speed_diff, 2) : 0;
			cost += collision_cost;
			
			// collision cost
			if (cost < lowest_cost)
			{
				lowest_cost = cost;
				best_state = state;
			}
//			cout << "state: " << state << " cost: " << cost << endl;
		}
		
		if (m_state != best_state && m_changeLaneTimer == 0)
		{
			cout << "state changed: " << best_state << endl;
			m_state = best_state;
			m_accTimer = 0;
			m_changeLaneTimer = 0;
		}
		timestep ++;
	}
	
	int getLaneForState(const string& state, const Car& car) const
	{
		if (state.compare("KL") == 0)
			return car.current_lane;
		else if (state.compare("LCL") == 0)
			return car.current_lane - 1;
		else
			return car.current_lane + 1;
	}
	
	double vehicleSpeed(const vector<double>& vehicle) const
	{
		double vx = vehicle[3];
		double vy = vehicle[4];
		return sqrt(vx*vx + vy*vy);
	}
	
	vector<double> getVehicle(const Car& car, int lane, bool front) const
	{
		vector<double> target_vehicle;
		double closest_distance = 999999;
		for (auto vehicle : car.sensor)
		{
			int vehicle_lane = vehicle[6] / kLanewidth + 1;
			double vehicle_s = vehicle[5];
			double distance = vehicle_s - car.s;
			
			bool predicate = front ? distance > 0 : distance < 0;
			if (vehicle_lane == lane && predicate)
			{
				if (abs(distance) < closest_distance)
				{
					closest_distance = distance;
					target_vehicle = vehicle;
				}
			}
		}
		return target_vehicle;
	}

	double generateNextJMTPoint(const vector<double>& start, const vector<double>& end, double T, double t)
	{
		vector<double> coeffs = JMT(start, end, T);
		return coeffs[0] + coeffs[1]*t + coeffs[2]*pow(t,2) + coeffs[3]*pow(t,3) + coeffs[4]*pow(t,4) + coeffs[5]*pow(t,5);
	}
	
	vector<double> realize_keep_lane(const MapInfo& mapInfo, const Car& car, const vector<int>& wayPointsIndexList, double current_s)
	{
		cout << "keep lane! " << "timestep " << timestep <<  endl;

		double x, y, next_s, next_v;
		double targetSpeed = kTargetSpeedMS;
		double timeToTarget = kMaxAccelerationTimeToTarget;

		bool too_close = false;
		static bool previous_too_close = too_close;
		vector<double> vehicle = getVehicle(car, car.current_lane, true);
		if (!vehicle.empty())
		{
			double frontCarSpeed = vehicleSpeed(vehicle);
			double distanceToFront = vehicle[5] - car.s;

			if (distanceToFront < 30)
			{
				too_close = true;
				targetSpeed = frontCarSpeed - 2;
				timeToTarget = 3;
			}
		}
		
		if (previous_too_close != too_close) // on change
		{
			previous_too_close = too_close;
			m_accTimer = 0;
		}
		double error = abs(m_current_v - targetSpeed);
		bool acc_activated = (error > kMinSpeedError) ? true : false;
		
		if (acc_activated) // activate adaptive cruise control
		{
			static double last_v = m_current_v;
			static double last_target_speed = targetSpeed;

			if (m_accTimer == 0)
			{
				last_v = m_current_v;
				last_target_speed = targetSpeed;
			}
			
			vector <double> s_start = {last_v, 0, 0};
			vector <double> s_end = {targetSpeed, 0, 0};
			m_accTimer ++;

			double t = m_accTimer * kControlInterval;
			next_v = generateNextJMTPoint(s_start, s_end, timeToTarget, t);
			cout << "acc timer " << m_accTimer << " t " << t << " next_v " << next_v << " error " << error << endl;
		}
		else
		{
			m_accTimer = 0;
			next_v = targetSpeed;
			cout << "constant speed" << endl;
		}
		const double dist_inc = next_v / kControlFrequency;
		next_s = current_s + dist_inc;
		
		double next_d = car.current_lane * kLanewidth - 2;
		m_current_d = next_d;
		m_current_v = next_v;
		tk::spline spline_sx, spline_sy;
		getSplineForWayPoints(mapInfo, wayPointsIndexList, spline_sx, spline_sy, next_d);
		
		x = spline_sx(next_s);
		y = spline_sy(next_s);
		return {x, y, next_s};
	}
	
	vector<double> realize_change_lane(const MapInfo& mapInfo, const Car& car, const vector<int>& wayPointsIndexList, double current_s, double target_d)
	{
		double x, y, next_s, next_d;
		double error = abs(m_current_d - target_d);
		bool change_lane = (error > kMinLaneError) ? true : false;
		
		if (change_lane)
		{
			static double last_s = m_current_s;
			static double last_d = m_current_d;
			static double last_v = m_current_v;

			if (m_changeLaneTimer == 0)
			{
				last_s = m_current_s;
				last_d = m_current_d;
				last_v = m_current_v;
			}
			
			double s_diff = kMaxChangeLaneTime * last_v;
			
			vector <double> s_start = {last_s, last_v, 0};
			vector <double> s_end = {last_s + s_diff, last_v, 0};
			vector <double> d_start = {last_d, 0, 0};
			vector <double> d_end = {target_d, 0, 0};
			
			m_changeLaneTimer ++;
			double t = m_changeLaneTimer * kControlInterval;
			next_s = generateNextJMTPoint(s_start, s_end, kMaxChangeLaneTime, t);
			next_d = generateNextJMTPoint(d_start, d_end, kMaxChangeLaneTime, t);
		}
		else
		{
			const double dist_inc = kTargetSpeedMS / kControlFrequency;
			next_s = current_s + dist_inc;
			next_d = car.current_lane * kLanewidth - 2;
			m_changeLaneTimer = 0;
			m_state = "KL";
		}
		
		cout << "m_changeLaneTimer " << m_changeLaneTimer << " next_s " << next_s << " next_d " << next_d << " error " << error  << " target_d " << target_d << endl;
		tk::spline spline_sx, spline_sy;
		getSplineForWayPoints(mapInfo, wayPointsIndexList, spline_sx, spline_sy, next_d);
		
		m_current_d = next_d;

		x = spline_sx(next_s);
		y = spline_sy(next_s);
		
		return {x, y, next_s};
	}
	
	vector<double> realize_change_left(const MapInfo& mapInfo, const Car& car, const vector<int>& wayPointsIndexList, double current_s)
	{
		cout << "lane change left!" << "timestep " << timestep << endl;
		if (m_changeLaneTimer == 0)
			m_target_d = (car.current_lane - 1) * kLanewidth - 2;
		
		return realize_change_lane(mapInfo, car, wayPointsIndexList, current_s, m_target_d);
	}
	
	vector<double> realize_change_right(const MapInfo& mapInfo, const Car& car, const vector<int>& wayPointsIndexList, double current_s)
	{
		cout << "lane change right!" << "timestep " << timestep << endl;
		if (m_changeLaneTimer == 0)
			m_target_d = (car.current_lane + 1) * kLanewidth - 2;
		
		return realize_change_lane(mapInfo, car, wayPointsIndexList, current_s, m_target_d);
	}
	
	vector<double> realize_next_state(const MapInfo& mapInfo, const Car& car, const vector<int>& wayPointsIndexList)
	{
		vector<double> result;
		if (m_state.compare("KL") == 0)
		{
			result = realize_keep_lane(mapInfo, car, wayPointsIndexList, m_current_s);
		}
		else if (m_state.compare("LCL") == 0)
		{
			result = realize_change_left(mapInfo, car, wayPointsIndexList, m_current_s);
		}
		else if (m_state.compare("LCR") == 0)
		{
			result = realize_change_right(mapInfo, car, wayPointsIndexList, m_current_s);
		}
		else
			assert(false);
		
		m_current_s = result[2];

		return result;
	}
	
private:
	string m_state = "KL";
	double m_target_d;
	int m_accTimer = 0;
	int m_changeLaneTimer = 0;
	double m_current_s = 0;
	double m_current_d = 0;
	double m_current_v = 0;
	int timestep = 0;
};

int main() {
	uWS::Hub h;
	
	// Load up map values for waypoint's x,y,s and d normalized normal vectors
	
	MapInfo mapInfo;
	
	// Waypoint map to read from
	string map_file_ = "../../data/highway_map.csv";
	
	ifstream in_map_(map_file_.c_str(), ifstream::in);
	
	string line;
	while (getline(in_map_, line)) {
		istringstream iss(line);
		double x;
		double y;
		float s;
		float d_x;
		float d_y;
		iss >> x;
		iss >> y;
		iss >> s;
		iss >> d_x;
		iss >> d_y;
		mapInfo.waypoints_x.push_back(x);
		mapInfo.waypoints_y.push_back(y);
		mapInfo.waypoints_s.push_back(s);
		mapInfo.waypoints_dx.push_back(d_x);
		mapInfo.waypoints_dy.push_back(d_y);
	}
	
	string log_file = "../../data/logger.csv";
	ofstream out_log(log_file.c_str(), ofstream::out);
	int lane = 2; // {left:1, middle:2, right:3}
	Car car;
	Planner planner;
	
	h.onMessage([&mapInfo, &car, &planner, &out_log, &lane](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length, uWS::OpCode opCode) {
		// "42" at the start of the message means there's a websocket message event.
		// The 4 signifies a websocket message
		// The 2 signifies a websocket event
		//auto sdata = string(data).substr(0, length);
		//cout << sdata << endl;
		if (length && length > 2 && data[0] == '4' && data[1] == '2') {
			
			auto s = hasData(data);
			
			if (s != "") {
				auto j = json::parse(s);
				
				string event = j[0].get<string>();
				
				if (event == "telemetry") {
					// j[1] is the data JSON object
					
					// Main car's localization Data
					car.x = j[1]["x"];
					car.y = j[1]["y"];
					car.s = j[1]["s"];
					car.d = j[1]["d"];
					car.speed = j[1]["speed"];
					double car_yaw = j[1]["yaw"];
					car.theta = deg2rad(car_yaw);
					car.current_lane = car.d / kLanewidth + 1;
					
					// Previous path data given to the Planner
					auto previous_path_x = j[1]["previous_path_x"];
					auto previous_path_y = j[1]["previous_path_y"];
					// Previous path's end s and d values
					double end_path_s = j[1]["end_path_s"];
					double end_path_d = j[1]["end_path_d"];
					
					// Sensor Fusion Data, a list of all other cars on the same side of the road.
					vector<vector<double>> sensor = j[1]["sensor_fusion"];
					car.sensor = sensor;
					/////////////////////////////////////////////////////////////////////////////////////////
					vector<int> wayPointsIndexList = findNextWayPointsIndexList(mapInfo, car);
					
					planner.updateState(car);
					
					////////////////////////////////////////////////////////////////////////////////////////////////////////////
					
					vector<double> next_x_vals, next_y_vals;
					double pos_x, pos_y;
					int path_size = previous_path_x.size();
					
					if(path_size == 0)
					{
						pos_x = car.x;
						pos_y = car.y;
						planner.init(car.s);
					}
					else
					{
						pos_x = previous_path_x[path_size-1];
						pos_y = previous_path_y[path_size-1];
					}
					
					for(int i = 0; i < path_size; i++)
					{
						next_x_vals.push_back(previous_path_x[i]);
						next_y_vals.push_back(previous_path_y[i]);
					}
					
					for(int i = 0; i < 50 - path_size; i++)
					{
						vector<double> next_point = planner.realize_next_state(mapInfo, car, wayPointsIndexList);
						double x = next_point[0];
						double y = next_point[1];
						double s = next_point[2];
						
						next_x_vals.push_back(x);
						next_y_vals.push_back(y);
						out_log << std::setprecision(9) << s << "," << x << "," << s << "," << y << endl;
						cout << std::setprecision(9) << "s: " << s << " x: " << x << " y: " << y << endl;
					}
					
					// send json message
					json msgJson;
					msgJson["next_x"] = next_x_vals;
					msgJson["next_y"] = next_y_vals;
					
					auto msg = "42[\"control\","+ msgJson.dump()+"]";
					ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
					
				}
			} else {
				// Manual driving
				std::string msg = "42[\"manual\",{}]";
				ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
			}
		}
	});
	
	// We don't need this since we're not using HTTP but if it's removed the
	// program
	// doesn't compile :-(
	h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
					   size_t, size_t) {
		const std::string s = "<h1>Hello world!</h1>";
		if (req.getUrl().valueLength == 1) {
			res->end(s.data(), s.length());
		} else {
			// i guess this should be done more gracefully?
			res->end(nullptr, 0);
		}
	});
	
	h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
		std::cout << "Connected!!!" << std::endl;
	});
	
	h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
						   char *message, size_t length) {
		ws.close();
		std::cout << "Disconnected" << std::endl;
	});
	
	int port = 4567;
	if (h.listen(port)) {
		std::cout << "Listening to port " << port << std::endl;
	} else {
		std::cerr << "Failed to listen to port" << std::endl;
		return -1;
	}
	h.run();
}
