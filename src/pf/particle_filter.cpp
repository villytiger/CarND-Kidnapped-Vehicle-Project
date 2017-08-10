/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <tuple>

#include <parallel/algorithm>

#include "particle_filter.h"

using namespace std;

namespace {

template <typename T1, typename T2>
double dist(const T1& a, const T2& b) {
  return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}
}

void ParticleFilter::init(double x, double y, double theta,
                          std::array<double, 3> std) {
  // TODO: Set the number of particles. Initialize all particles to first
  // position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and
  // others in this file).

  default_random_engine gen;

  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i != 10; ++i)
    particles_.emplace_back(
        Particle{i, dist_x(gen), dist_y(gen), dist_theta(gen), 1});

  is_initialized_ = true;
}

void ParticleFilter::prediction(double delta_t, std::array<double, 3> std_pos,
                                double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and
  // std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/

  default_random_engine gen;

  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  for (auto& p : particles_) {
    if (yaw_rate) {
      p.x += velocity * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta)) /
             yaw_rate;
      p.y += velocity * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t)) /
             yaw_rate;
      p.theta += yaw_rate * delta_t;
    } else {
      p.x += velocity * delta_t * cos(p.theta);
      p.x += velocity * delta_t * sin(p.theta);
    }

    p.x += dist_x(gen);
    p.y += dist_y(gen);
    p.theta += dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(const std::vector<LandmarkObs>& predicted,
                                     std::vector<LandmarkObs>* observations) {
  // TODO: Find the predicted measurement that is closest to each observed
  // measurement and assign the
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will
  // probably find it useful to
  //   implement this method and use it as a helper during the updateWeights
  //   phase.
}

void ParticleFilter::updateWeights(double sensor_range,
                                   std::array<double, 2> std_landmark,
                                   const std::vector<LandmarkObs>& observations,
                                   const Map& map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian
  // distribution. You can read
  //   more about this distribution here:
  //   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your
  // particles are located
  //   according to the MAP'S coordinate system. You will need to transform
  //   between the two systems.
  //   Keep in mind that this transformation requires both rotation AND
  //   translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement
  //   (look at equation
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html

  std::vector<LandmarkObs> observed_landmarks;
  std::vector<LandmarkObs> possible_landmarks;

  for (auto& p : particles_) {
    p.associations.clear();
    p.sense_x.clear();
    p.sense_y.clear();

    possible_landmarks.clear();
    for (const auto& lm : map_landmarks.landmark_list)
      if (dist(p, lm) < sensor_range)
        possible_landmarks.emplace_back(LandmarkObs{lm.id, lm.x, lm.y});

    if (possible_landmarks.empty()) continue;

    observed_landmarks = observations;
    for (auto& o : observed_landmarks) {
      tie(o.x, o.y) = make_pair(p.x + cos(p.theta) * o.x - sin(p.theta) * o.y,
                                p.y + sin(p.theta) * o.x + cos(p.theta) * o.y);

      auto it =
          min_element(possible_landmarks.begin(), possible_landmarks.end(),
                      [&o](const auto& a, const auto& b) {
                        return dist(o, a) < dist(o, b);
                      });
      assert(it != possible_landmarks.end());

      o.id = it->id;
      p.associations.emplace_back(it->id);
      p.sense_x.emplace_back(o.x);
      p.sense_y.emplace_back(o.y);
    }

    p.weight = accumulate(
        observed_landmarks.begin(), observed_landmarks.end(), 1.,
        [&map_landmarks, std_landmark](double w, const auto& o) {
          const auto& ml = map_landmarks.landmark_list[o.id - 1];
          w *= exp(-pow(ml.x - o.x, 2) / (2. * pow(std_landmark[0], 2)) -
                   pow(ml.y - o.y, 2) / (2. * pow(std_landmark[1], 2)));
          w /= 2. * M_PI * std_landmark[0] * std_landmark[1];
          return w;
        });
  }
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to
  // their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  vector<double> weights;
  transform(particles_.begin(), particles_.end(), back_inserter(weights),
            [](const auto& p) { return p.weight; });

  default_random_engine gen;
  discrete_distribution<size_t> dist(weights.begin(), weights.end());

  decltype(particles_) particles(particles_.size());
  for (auto& p : particles) p = particles_[dist(gen)];

  particles_ = particles;
}

void ParticleFilter::SetAssociations(Particle* particle,
                                     const std::vector<int>& associations,
                                     const std::vector<double>& sense_x,
                                     const std::vector<double>& sense_y) {
  // particle: the particle to assign each listed association, and association's
  // (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  // Clear the previous associations
  particle->associations.clear();
  particle->sense_x.clear();
  particle->sense_y.clear();

  particle->associations = associations;
  particle->sense_x = sense_x;
  particle->sense_y = sense_y;
}

string ParticleFilter::getAssociations(const Particle& best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseX(const Particle& best) {
  vector<double> v = best.sense_x;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseY(const Particle& best) {
  vector<double> v = best.sense_y;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
