#ifndef SEARCH_H
#define SEARCH_H
#include "nn.h"
#include <iostream>
#include <set>
#include <sstream>
#include <unordered_set>
#include <vector>

using namespace std;

inline double LOO_accuracy(vector<vector<vector<double>>> &cache,
                           set<int> feature_list, vector<int> &labels) {
  int correct_count = 0;
  for (unsigned int i = 0; i < labels.size(); i++) {
    int label = query(cache, i, feature_list, labels);
    if (label == labels.at(i)) {
      correct_count++;
    } else {
    }
  }

  return correct_count / double(labels.size());
}

inline void forward(vector<vector<vector<double>>> &cache,
                    vector<int> &labels) {
  set<int> features;
  set<int> best_features;
  double best_acc_final = 0;
  unsigned int num_features = cache.at(0).at(0).size();

  while (features.size() != num_features) {
    double best_acc_batch = 0;
    int best_feature = -1;
    for (unsigned int i = 0; i < num_features; i++) {
      if (features.find(i) != features.end()) {
        continue;
      }

      set<int> try_features = features;
      try_features.insert(i);

      cout << "\tUsing feature set ";
      for (auto j : try_features) {
        cout << j + 1 << ' ';
      }
      cout << "..." << endl;

      double curr_acc = LOO_accuracy(cache, try_features, labels);
      cout << "\t\tAccuracy was " << curr_acc * 100 << "%" << endl;
      if (curr_acc > best_acc_batch) {
        best_acc_batch = curr_acc;
        best_feature = i;
      }
    }
    if (best_acc_batch > best_acc_final) {
      best_acc_final = best_acc_batch;
      features.insert(best_feature);
      best_features = features;
      cout << "Best feature set so far: ";
      for (auto i : best_features) {
        cout << i + 1 << ' ';
      }
      cout << "with accuracy " << best_acc_final * 100 << "%" << endl;
    } else {
      features.insert(best_feature);
      cout << "Best feature set this batch: ";
      for (auto i : features) {
        cout << i + 1 << ' ';
      }
      cout << "with accuracy " << best_acc_batch * 100 << "% instead of "
           << best_acc_final * 100 << "%" << endl;
      cout << "WARNING: accuracy has decreased. Continuing for completeness"
           << endl;
    }
  }
  cout << "Final feature set: ";
  for (auto i : best_features) {
    cout << i + 1 << ' ';
  }
  cout << "with accuracy " << best_acc_final * 100 << "%" << endl;

  t_end = chrono::high_resolution_clock::now();
  elapsed_time_ms =
      chrono::duration<double, std::milli>(t_end - t_start).count();
  cout << "Took " << elapsed_time_ms << " milliseconds to run forward search"
       << endl;
}

}

#endif
