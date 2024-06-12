#ifndef READ_H
#define READ_H

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

template <typename T> void print(T iterable, string sep, string end) {

  for (auto i : iterable) {
    cout << i << sep;
  }
  cout << end;
}

inline pair<int, vector<double>> split_lines(string &s) {
  int label;
  vector<double> features;
  stringstream ss(s);

  string curr;

  ss >> curr;
  label = stoi(curr);

  while (ss >> curr) {
    features.push_back(stod(curr));
  }

  return make_pair(label, features);
}

inline pair<vector<int>, vector<vector<double>>> read_file(string &filename) {
  auto t_start = chrono::high_resolution_clock::now();
  auto t_end = chrono::high_resolution_clock::now();
  double elapsed_time_ms;
  t_start = chrono::high_resolution_clock::now();

  vector<int> labels;
  vector<vector<double>> features;

  fstream fs(filename);

  if (!fs.is_open()) {
    throw runtime_error("Error opening path " + filename);
  }

  string line;
  while (getline(fs, line)) {

    pair<int, vector<double>> curr_sample = split_lines(line);
    labels.push_back(curr_sample.first);
    features.push_back(curr_sample.second);
  }

  t_end = chrono::high_resolution_clock::now();
  elapsed_time_ms = chrono::duration<double, milli>(t_end - t_start).count();
  cout << "Took " << elapsed_time_ms << " milliseconds to read file" << endl;

  return make_pair(labels, features);
}

#endif
