#include <chrono>
#include <cmath>
#include <iostream>

#include <uWS/uWS.h>

#include "json/json.hpp"

#include "particle_filter.h"

using std::cerr;
using std::cout;
using std::endl;
using std::string;
using std::vector;

using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::microseconds;

// for convenience
using json = nlohmann::json;

struct State {
  Map map;
  ParticleFilter pf;
};

// Time elapsed between measurements [sec]
constexpr const double kDeltaT = 0.1;

// Sensor range [m]
constexpr double kSensorRange = 50;

// GPS measurement uncertainty [x [m], y [m], theta [rad]]
constexpr const std::array<double, 3> kSigmaPos = {0.3, 0.3, 0.01};

// Landmark measurement uncertainty [x [m], y [m]]
constexpr const std::array<double, 2> kSigmaLandmark = {0.3, 0.3};

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(const string& s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_first_of("]");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 1);
  }
  return "";
}

int main() {
  uWS::Hub h;

  // Read map data
  Map map;
  if (!read_map_data("../data/map_data.txt", map)) {
    cout << "Error: Could not open map file" << endl;
    return -1;
  }

  h.onMessage([](uWS::WebSocket<uWS::SERVER> ws, char* data, size_t length,
                 uWS::OpCode opCode) {
    State* state = static_cast<State*>(ws.getUserData());

    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    if (!length || length <= 2 || data[0] != '4' || data[1] != '2') return;

    auto s = hasData(string(data));
    if (s.empty()) {
      string msg = "42[\"manual\",{}]";
      ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      return;
    }

    auto j = json::parse(s);

    string event = j[0].get<string>();

    if (event != "telemetry") return;

    auto start_time = high_resolution_clock::now();

    if (!state->pf.initialized()) {
      // Sense noisy position data from the simulator
      double sense_x = std::stod(j[1]["sense_x"].get<std::string>());
      double sense_y = std::stod(j[1]["sense_y"].get<std::string>());
      double sense_theta = std::stod(j[1]["sense_theta"].get<std::string>());

      state->pf.init(sense_x, sense_y, sense_theta, kSigmaPos);
    } else {
      // Predict the vehicle's next state from previous (noiseless
      // control) data.
      double previous_velocity =
          std::stod(j[1]["previous_velocity"].get<std::string>());
      double previous_yawrate =
          std::stod(j[1]["previous_yawrate"].get<std::string>());

      state->pf.prediction(kDeltaT, kSigmaPos, previous_velocity,
                           previous_yawrate);
    }

    auto end_time = high_resolution_clock::now();
    cout << "prediction time: "
         << duration_cast<microseconds>(end_time - start_time).count() << endl;

    // receive noisy observation data from the simulator
    // sense_observations in JSON format
    // [{obs_x,obs_y},{obs_x,obs_y},...{obs_x,obs_y}]
    vector<LandmarkObs> noisy_observations;
    string sense_observations_x = j[1]["sense_observations_x"];
    string sense_observations_y = j[1]["sense_observations_y"];

    std::vector<float> x_sense;
    std::istringstream iss_x(sense_observations_x);

    std::copy(std::istream_iterator<float>(iss_x),
              std::istream_iterator<float>(), std::back_inserter(x_sense));

    std::vector<float> y_sense;
    std::istringstream iss_y(sense_observations_y);

    std::copy(std::istream_iterator<float>(iss_y),
              std::istream_iterator<float>(), std::back_inserter(y_sense));

    for (int i = 0; i < x_sense.size(); i++) {
      LandmarkObs obs;
      obs.x = x_sense[i];
      obs.y = y_sense[i];
      noisy_observations.push_back(obs);
    }

    start_time = high_resolution_clock::now();

    // Update the weights and resample
    state->pf.updateWeights(kSensorRange, kSigmaLandmark, noisy_observations,
                            state->map);

    end_time = high_resolution_clock::now();
    cout << "update time: "
         << duration_cast<microseconds>(end_time - start_time).count() << endl;

    start_time = high_resolution_clock::now();

    state->pf.resample();

    end_time = high_resolution_clock::now();
    cout << "resample time: "
         << duration_cast<microseconds>(end_time - start_time).count() << endl;

    // Calculate and output the average weighted error of the particle
    // filter over all time steps so far.
    vector<Particle> particles = state->pf.Particles();
    int num_particles = particles.size();
    double highest_weight = -1.0;
    Particle best_particle;
    double weight_sum = 0.0;
    for (int i = 0; i < num_particles; ++i) {
      if (particles[i].weight > highest_weight) {
        highest_weight = particles[i].weight;
        best_particle = particles[i];
      }
      weight_sum += particles[i].weight;
    }

    cout << "highest w " << highest_weight << endl;
    cout << "average w " << weight_sum / num_particles << endl;

    json msgJson;
    msgJson["best_particle_x"] = best_particle.x;
    msgJson["best_particle_y"] = best_particle.y;
    msgJson["best_particle_theta"] = best_particle.theta;

    // Optional message data used for debugging particle's sensing and
    // associations
    msgJson["best_particle_associations"] =
        state->pf.getAssociations(best_particle);
    msgJson["best_particle_sense_x"] = state->pf.getSenseX(best_particle);
    msgJson["best_particle_sense_y"] = state->pf.getSenseY(best_particle);

    auto msg = "42[\"best_particle\"," + msgJson.dump() + "]";
    // std::cout << msg << std::endl;
    ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse* res, uWS::HttpRequest req, char* data,
                     size_t, size_t) {
    const string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    Map map;
    if (!read_map_data("../data/map_data.txt", map))
      cerr << "Error: Could not open map file" << endl;

    ws.setUserData(new State{std::move(map)});
    cout << "Connected!!!" << endl;
  });

  h.onDisconnection([](uWS::WebSocket<uWS::SERVER> ws, int code, char* message,
                       size_t length) {
    delete static_cast<State*>(ws.getUserData());
    cout << "Disconnected" << endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    cout << "Listening to port " << port << endl;
  } else {
    cerr << "Failed to listen to port" << endl;
    return -1;
  }
  h.run();
}
