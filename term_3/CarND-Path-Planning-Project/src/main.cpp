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

struct Point
{
	double x;
	double y;
};

Point worldToBody(double x, double y, double theta, double px, double py)
{
	double x_b = px - x;
	double y_b = py - y;
	double px_b = x_b * cos(theta) + y_b * sin(theta);
	double py_b = -x_b * sin(theta) + y_b * cos(theta);
	return {px_b, py_b};
}

Point bodyToWorld(double x, double y, double theta, double px_b, double py_b)
{
	double px = x + px_b * cos(theta) - py_b * sin(theta);
	double py = y + px_b * sin(theta) + py_b * cos(theta);
	return {px, py};
}

float rampingUp(int timeStep)
{
	return 22 / (1 + exp(-0.012 * (timeStep - 300))); // logistic function
}

double changeRight(int timeStep)
{
//	return 6 + 4 / (1 + exp(-0.02 * (timeStep - 500))); // logistic function
	return (4.0/400)*timeStep + 6;
}


double getWayPointSegmentDistance(const vector<int>& wayPointsIndex, const vector<double>& map_waypoints_s)
{
	double distance;
	
	// check if the car has wrapped around by looking at the position of 0(start waypoint index) in our waypoint segment.
	int index = std::find(wayPointsIndex.begin(), wayPointsIndex.end(), 0) - wayPointsIndex.begin();
	if (index != wayPointsIndex.size() && index != 0)
	{
		// use the distance between second last waypoint and the last.
		const double kGapLastAndFirst = 42.6;
		double startPortion = (index == 1) ? 0 : (map_waypoints_s[wayPointsIndex[index-1]] - map_waypoints_s[wayPointsIndex.front()]);
		double endPortion = (index == (wayPointsIndex.size() - 1)) ? 0 : (map_waypoints_s[wayPointsIndex.back()] - map_waypoints_s[0]);
		distance = startPortion + kGapLastAndFirst + endPortion;
	}
	else
		distance = map_waypoints_s[wayPointsIndex.back()] - map_waypoints_s[wayPointsIndex.front()];
	
	return distance;
}

vector<double> JMT(vector< double> start, vector <double> end, double T)
{
	/*
	 Calculate the Jerk Minimizing Trajectory that connects the initial state
	 to the final state in time T.
	 
	 INPUTS
	 
	 start - the vehicles start location given as a length three array
	 corresponding to initial values of [s, s_dot, s_double_dot]
	 
	 end   - the desired end state for vehicle. Like "start" this is a
	 length three array.
	 
	 T     - The duration, in seconds, over which this maneuver should occur.
	 
	 OUTPUT
	 an array of length 6, each value corresponding to a coefficent in the polynomial
	 s(t) = a_0 + a_1 * t + a_2 * t**2 + a_3 * t**3 + a_4 * t**4 + a_5 * t**5
	 
	 EXAMPLE
	 
	 > JMT( [0, 10, 0], [10, 10, 0], 1)
	 [0.0, 10.0, 0.0, 0.0, 0.0, 0.0]
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
	{
		result.push_back(C.data()[i]);
	}
	
	return result;
}


int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0
  double max_s = 6945.554;

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
  	map_waypoints_x.push_back(x);
  	map_waypoints_y.push_back(y);
  	map_waypoints_s.push_back(s);
  	map_waypoints_dx.push_back(d_x);
  	map_waypoints_dy.push_back(d_y);
  }
  namespace plt = matplotlibcpp;
	
  string log_file = "../../data/logger.csv";
  ofstream out_log(log_file.c_str(), ofstream::out);
  int rampUpTimer = 0;

  h.onMessage([&map_waypoints_x,&map_waypoints_y,&map_waypoints_s,&map_waypoints_dx,&map_waypoints_dy, &out_log, &rampUpTimer](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length, uWS::OpCode opCode) {
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
          	double car_x = j[1]["x"];
          	double car_y = j[1]["y"];
          	double car_s = j[1]["s"];
          	double car_d = j[1]["d"];
          	double car_yaw = j[1]["yaw"];
          	double car_speed = j[1]["speed"];
			double theta = deg2rad(car_yaw);

          	// Previous path data given to the Planner
          	auto previous_path_x = j[1]["previous_path_x"];
          	auto previous_path_y = j[1]["previous_path_y"];
          	// Previous path's end s and d values 
          	double end_path_s = j[1]["end_path_s"];
          	double end_path_d = j[1]["end_path_d"];

          	// Sensor Fusion Data, a list of all other cars on the same side of the road.
          	auto sensor_fusion = j[1]["sensor_fusion"];
//			cout << "car pos " << car_x << " " << car_y << " " << car_yaw << endl;

			// find next waypoints list
			const int kMaxNextWayPoints = 10;
			vector<double> wayPoints_x, wayPoints_y;
			int nextWayPoint = ClosestWaypoint(car_x, car_y, map_waypoints_x, map_waypoints_y); // the first waypoint index of a waypoints segment
			cout << "closet waypoint index " << nextWayPoint << endl;
			nextWayPoint = nextWayPoint - 4;
			static int previousIndex = nextWayPoint;
			static double car_next_x = car_x;
			static double car_next_y = car_y;
			static double next_theta = theta;
			if (nextWayPoint != previousIndex)
			{
				previousIndex = nextWayPoint;
				car_next_x = car_x;
				car_next_y = car_y;
				next_theta = theta;
			}

			if (nextWayPoint < 0)
				nextWayPoint = map_waypoints_x.size() + nextWayPoint;
			
			vector<int> wayPointsIndex;
			for (int i = 0, nextwayPointIndex = nextWayPoint; i < kMaxNextWayPoints; i++)
			{
				wayPointsIndex.push_back(nextwayPointIndex);
				wayPoints_x.push_back(map_waypoints_x[nextwayPointIndex]);
				wayPoints_y.push_back(map_waypoints_y[nextwayPointIndex]);
				nextwayPointIndex++;
				nextwayPointIndex = nextwayPointIndex % (map_waypoints_x.size());
			}
			cout << "wayPointsIndex: ";
			for (int i = 0; i < wayPointsIndex.size(); i++)
				cout << " " << wayPointsIndex[i];
			cout << endl;
			
			
			// prepare 50 points to send
			vector<double> next_x_vals, next_y_vals;
			double pos_x, pos_y;
			int path_size = previous_path_x.size();
			
			for(int i = 0; i < path_size; i++)
			{
				next_x_vals.push_back(previous_path_x[i]);
				next_y_vals.push_back(previous_path_y[i]);
			}
			
			if(path_size == 0)
			{
				pos_x = car_x;
				pos_y = car_y;
			}
			else
			{
				pos_x = previous_path_x[path_size-1];
				pos_y = previous_path_y[path_size-1];
			}

			
			double lane = 6;
			static bool change_lane = false;
			vector<double> xy = {0,0};
			
			if (rampUpTimer >= 400 && rampUpTimer < 550)
				change_lane = true;
			if (rampUpTimer >= 550)
			{
				lane = 10;
				change_lane = false;
			}
			if (change_lane)
			{
				static vector<double> s_coeffs, d_coeffs;
				static bool generated = false;
				static vector<double> wayPoints_x, wayPoints_y, wayPoints_s;

				if (!generated)
				{
					vector<double> lane_waypoints_x, lane_waypoints_y, 	lane_waypoints_s;
;
					for(auto index : wayPointsIndex)
					{
						lane_waypoints_x.push_back(map_waypoints_x[index]);
						lane_waypoints_y.push_back(map_waypoints_y[index]);
						lane_waypoints_s.push_back(map_waypoints_s[index]);
					}
					
					// transform waypoints from world to body frame
					vector<double> wayPoints_bx, wayPoints_by;
					for (int i = 0; i < lane_waypoints_x.size(); i++)
					{
						Point point_b = worldToBody(car_next_x, car_next_y, next_theta, lane_waypoints_x[i], lane_waypoints_y[i]);
						wayPoints_bx.push_back(point_b.x);
						wayPoints_by.push_back(point_b.y);
					}
					
					// build spline for interplotation
					tk::spline s;
					s.set_points(wayPoints_bx, wayPoints_by);
					
					tk::spline s_s;
					s_s.set_points(wayPoints_bx, lane_waypoints_s);
					double x_step = (wayPoints_bx.back() - wayPoints_bx.front()) / 150.0;
					//double s_step = (map_waypoints_s[wayPointsIndex.back()] - map_waypoints_s[wayPointsIndex.front()]) / 100.0;

					for(int i = 0; i < 50; i++)
					{
						const double xb = wayPoints_bx.front() + i * x_step;
						const double yb = s(xb);
						
						Point point = bodyToWorld(car_next_x, car_next_y, next_theta, xb, yb);
						wayPoints_x.push_back(point.x);
						wayPoints_y.push_back(point.y);
						wayPoints_s.push_back(s_s(xb));
//						out_log << std::setprecision(9) << point.x << "," << point.y << "," << wayPoints_s[i] << endl;
					}
					
					vector <double> s_start = {end_path_s, car_speed, 0};
					vector <double> s_end = {end_path_s + 75, car_speed, 0};
					
					vector <double> d_start = {end_path_d, 0, 0};
					vector <double> d_end = {10, 0, 0};

					s_coeffs = JMT(s_start, s_end, 3);
					d_coeffs = JMT(d_start, d_end, 3);
					generated = true;
				}
				
				static int lastPos = rampUpTimer;
				for (int i = 0; i < 50-path_size; i++)
				{
					double t = (lastPos - 400) *  (3.0 / 150.0);
					double s_next = s_coeffs[0] + s_coeffs[1]*t + s_coeffs[2]*pow(t,2) + s_coeffs[3]*pow(t,3) + s_coeffs[4]*pow(t,4) + s_coeffs[5]*pow(t,5);
					double d_next = d_coeffs[0] + d_coeffs[1]*t + d_coeffs[2]*pow(t,2) + d_coeffs[3]*pow(t,3) + d_coeffs[4]*pow(t,4) + d_coeffs[5]*pow(t,5);
					
					cout << "!!!!!!!!" << "speed " << car_speed << "," << t << "," << s_next << "," << d_next << endl;
					xy = getXY(s_next, d_next, wayPoints_s, wayPoints_x, wayPoints_y);
//					xy = getXY(s_next, d_next, map_waypoints_s, map_waypoints_x, map_waypoints_y);

//					out_log << std::setprecision(9) << t << "," << s_next << "," << t << "," << d_next << endl;
//					out_log << std::setprecision(9) << t << "," << s_next << "," << xy[0] << "," << xy[1] << endl;

					next_x_vals.push_back(xy[0]);
					next_y_vals.push_back(xy[1]);
					out_log << std::setprecision(9) <<  xy[0] << "," << xy[1]  << endl;
					lastPos ++;
				}

			}
			
			cout << xy[0] << "," << xy[1] << "end_s" << end_path_s << "car_s " << car_s << " car_d" << car_d << "time step " << rampUpTimer++ << endl;

			if (!change_lane)
			{
				// choose lane, middle lane for now
				vector<double> lane_waypoints_x, lane_waypoints_y;
				for(auto index : wayPointsIndex)
				{
					lane_waypoints_x.push_back(map_waypoints_x[index] + lane * map_waypoints_dx[index]);
					lane_waypoints_y.push_back(map_waypoints_y[index] + lane * map_waypoints_dy[index]);
				}
				

				// transform waypoints from world to body frame
				vector<double> wayPoints_bx, wayPoints_by;
				for (int i = 0; i < lane_waypoints_x.size(); i++)
				{
					Point point_b = worldToBody(car_next_x, car_next_y, next_theta, lane_waypoints_x[i], lane_waypoints_y[i]);
					wayPoints_bx.push_back(point_b.x);
					wayPoints_by.push_back(point_b.y);
				}

				// build spline for interplotation
				tk::spline s;
				s.set_points(wayPoints_bx, wayPoints_by);
				
				// ramping up smoothly using s curve
				const float kSpeedLimit = rampingUp(rampUpTimer);
	//			const float kSpeedLimit = 22; // m/s
				const int kControllerFrequency = 50; //50Hz, car moves 50 times per second
				// use car frenet?
				double	distance = getWayPointSegmentDistance(wayPointsIndex, map_waypoints_s);
				// calculate sampling frequency base on speed
				const int samplingFrequency = distance * kControllerFrequency / kSpeedLimit;
				
				
				const double x_step = (wayPoints_bx.back() - wayPoints_bx.front()) / samplingFrequency;
				
				cout << "path_size" << path_size << "last point to be: " << previous_path_x[0] << "," << previous_path_y[0] << endl;
				cout << "final point: ";

				Point point_b = worldToBody(car_next_x, car_next_y, next_theta, pos_x, pos_y);
				
				for(int i = 1; i <= 50-path_size; i++)
				{
					const double xb = point_b.x + i * x_step;
					const double yb = s(xb);
					
					Point point = bodyToWorld(car_next_x, car_next_y, next_theta, xb, yb);
					next_x_vals.push_back(point.x);
					next_y_vals.push_back(point.y);
					cout << "(" << point.x << "," << point.y << ")," ;
//					out_log << std::setprecision(9) << point.x << "," << point.y << endl;
//					<< car_x << "," << car_y << "," << car_yaw << endl;
					out_log << std::setprecision(9) << point.x << "," << point.y  << endl;

				}
				cout << endl;
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


