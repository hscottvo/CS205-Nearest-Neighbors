#ifndef NN_H
#define NN_H

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <execution>
#include <iostream>
#include <math.h>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <vector>

using namespace std;

inline double default_rate(vector<int> labels) {
  unordered_map<int, int> map;
  for (auto i : labels) {
    auto element = map.find(i);
    if (element == map.end()) {
      map.insert(make_pair(i, 1));
    } else {
      element->second++;
    }
  }

  int max_element = 0;
  for (auto i : map) {
    max_element = max(max_element, i.second);
  }

  return max_element / double(labels.size());
}

inline vector<vector<vector<double>>>
instantiate_cache(vector<vector<double>> &data) {
  auto t_start = chrono::high_resolution_clock::now();
  auto t_end = chrono::high_resolution_clock::now();
  double elapsed_time_ms;
  t_start = chrono::high_resolution_clock::now();

  vector<vector<vector<double>>> ret(
      data.size(), vector<vector<double>>(
                       data.size(), vector<double>(data.at(0).size(), -1)));
  t_end = chrono::high_resolution_clock::now();
  elapsed_time_ms = chrono::duration<double, milli>(t_end - t_start).count();
  cout << "Took " << elapsed_time_ms
       << " milliseconds to instantiate cache array of shape ";
  cout << ret.size() << ", " << ret.at(0).size() << ", " << ret[0][0].size()
       << endl;
  return ret;
}

inline vector<vector<vector<double>>>
cache_diff_squared(vector<vector<double>> &data) {
  auto t_start = chrono::high_resolution_clock::now();
  auto t_end = chrono::high_resolution_clock::now();
  double elapsed_time_ms;

  auto ret = instantiate_cache(data);

  t_start = chrono::high_resolution_clock::now();

  vector<int> i_idxs(data.size(), -1);
  unsigned int j;
  unsigned int k;
  iota(i_idxs.begin(), i_idxs.end(), 0);

  for (unsigned int i = 0; i < data.size(); i++) {
    // for_each(execution::par, i_idxs.begin(), i_idxs.end(),
    //          [&data, &j, &k, &ret](int &i) {
    for (j = i + 1; j < data.size(); j++) {
      for (k = 0; k < data[0].size(); k++) {
        ret[i][j][k] = pow(data[i][k] - data[j][k], 2);
      }
    }
  }
  t_end = chrono::high_resolution_clock::now();
  elapsed_time_ms = chrono::duration<double, milli>(t_end - t_start).count();
  cout << "Took " << elapsed_time_ms << " milliseconds to fill cache file"
       << endl;

  return ret;
}

inline int query(vector<vector<vector<double>>> &cache, unsigned int index,
                 set<int> &feature_list, vector<int> &labels) {

  if (index >= cache.size()) {
    string s;
    stringstream ss;
    ss << "Expected from 0 to " << cache.size() << ". Got " << index << endl;
    ss >> s;
    throw out_of_range(s);
  }

  // https://stackoverflow.com/questions/5834635/how-do-i-get-double-max
  double min_distance = DBL_MAX;
  int min_index = -1;

  for (unsigned int i = 0; i < index; i++) {
    double sample_distance = 0;
    for (int feature : feature_list) {
      if (abs(cache.at(i).at(index).at(feature) - -1) < .0001) {
        exit(1);
      }
      sample_distance += cache.at(i).at(index).at(feature);
    }

    if (min_distance > sample_distance) {
      min_distance = sample_distance;
      min_index = i;
    }
  }

  for (unsigned int i = index + 1; i < cache.size(); i++) {
    double sample_distance = 0;
    for (int feature : feature_list) {
      if (abs(cache.at(index).at(i).at(feature) - -1) < .0001) {
        exit(1);
      }

      sample_distance += cache.at(index).at(i).at(feature);
    }

    if (min_distance > sample_distance) {
      min_distance = sample_distance;
      min_index = i;
    }
  }

  if (min_index != -1) {
    return labels.at(min_index);
  } else {
    cout << "This should never happen" << endl;
    exit(1);
  }
}

#endif
