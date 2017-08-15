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

const int kLanewidth = 4;

struct Car
{
	enum State
	{
		kKeepLane,
		kChangeLeft,
		kChangeRight,
	};
	
	double x;
	double y;
	double s;
	double d;
	double speed;
	double theta;
	State state;
	int current_lane;
};

struct MapInfo
{
	vector<double> waypoints_x;
	vector<double> waypoints_y;
	vector<double> waypoints_s;
	vector<double> waypoints_dx;
	vector<double> waypoints_dy;
};

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

double changeLane(int time, int current_lane, bool right)
{
	int k = right ? 1 : -1;
	return current_lane + k / (1 + exp(-0.03 * (time - 190))); // logistic function
}

double getNextS(int timestep, double current_s)
{
	double next_s;
	const double dist_inc = 0.43;

	if (timestep < 200)
	{
		vector <double> s_start = {124.8336, 0, 0};
		vector <double> s_end = {124.8336 + 43, 21.5, 0};
		
		vector<double> s_coeffs = JMT(s_start, s_end, 4);
		
		double t = timestep * 0.02;
		next_s = s_coeffs[0] + s_coeffs[1]*t + s_coeffs[2]*pow(t,2) + s_coeffs[3]*pow(t,3) + s_coeffs[4]*pow(t,4) + s_coeffs[5]*pow(t,5);
	}
	else
		next_s = current_s + dist_inc;
	
	return next_s;
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

Car::State getNextState(const vector<vector<double>>& sensor, const Car& car)
{
	Car::State state;
	
	return state;
}

vector<double> realize_keep_lane(const MapInfo& mapInfo, const Car& car, const vector<int>& wayPointsIndexList, int timestep, double current_point_s)
{
	double x, y, next_s;
	double next_d = car.current_lane * kLanewidth - 2;
	
	tk::spline spline_sx, spline_sy;
	getSplineForWayPoints(mapInfo, wayPointsIndexList, spline_sx, spline_sy, next_d);
	
	if (timestep < 200)
	{
		vector <double> s_start = {124.8336, 0, 0};
		vector <double> s_end = {124.8336 + 43, 21.5, 0};
		
		vector<double> s_coeffs = JMT(s_start, s_end, 4);
		
		double t = timestep * 0.02;
		next_s = s_coeffs[0] + s_coeffs[1]*t + s_coeffs[2]*pow(t,2) + s_coeffs[3]*pow(t,3) + s_coeffs[4]*pow(t,4) + s_coeffs[5]*pow(t,5);
	}
	else
	{
		const double dist_inc = 0.43; //21.5m/s, 48mph
		next_s = current_point_s + dist_inc;
	}
	
	x = spline_sx(next_s);
	y = spline_sy(next_s);
	
	return {x, y, next_s};
}

vector<double> realize_change_left(const Car& car, double current_point_s)
{
	double x, y, next_s;
	
	return {x, y, next_s};
}

vector<double> realize_change_right(const Car& car, double current_point_s)
{
	double x, y, next_s;
	
	return {x, y, next_s};
}

vector<double> realize_next_state(const MapInfo& mapInfo, const Car& car, const vector<int>& wayPointsIndexList, int points_scheduled, double current_point_s)
{
	switch (car.state)
	{
		case Car::kKeepLane:
			return realize_keep_lane(mapInfo, car, wayPointsIndexList, points_scheduled, current_point_s);
			
		case Car::kChangeLeft:
			return realize_change_left(car, current_point_s);
			
		case Car::kChangeRight:
			return realize_change_right(car, current_point_s);

		default:
			assert(false);
	}
}

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
  int timestep = 0;
  int lane = 2; // {left:1, middle:2, right:3}
  int ref_speed;
  Car car = {0};
	
  h.onMessage([&mapInfo, &car, &out_log, &timestep, &ref_speed, &lane](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length, uWS::OpCode opCode) {
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
          	auto sensor_fusion = j[1]["sensor_fusion"];
			
/////////////////////////////////////////////////////////////////////////////////////////
			vector<int> wayPointsIndexList = findNextWayPointsIndexList(mapInfo, car);
			timestep ++;

			// Behaviour
			car.state = getNextState(sensor_fusion, car);
		
////////////////////////////////////////////////////////////////////////////////////////////////////////////

			vector<double> next_x_vals, next_y_vals;
			double pos_x, pos_y;
			int path_size = previous_path_x.size();
			static int points_scheduled = 0;
			static double next_s = car.s;

			if(path_size == 0)
			{
				pos_x = car.x;
				pos_y = car.y;
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
				points_scheduled++;
				vector<double> next_point = realize_next_state(mapInfo, car, wayPointsIndexList, points_scheduled, next_s);
				double x = next_point[0];
				double y = next_point[1];
				next_s += next_point[2];
				
				next_x_vals.push_back(x);
				next_y_vals.push_back(y);
				out_log << std::setprecision(9) << next_s << "," << x << "," << next_s << "," << y << endl;
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


