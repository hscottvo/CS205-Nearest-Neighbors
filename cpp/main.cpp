#include "nn.h"
#include "read.h"
#include "search.h"
// https://stackoverflow.com/questions/728068/how-to-calculate-a-time-difference-in-c
#include <string>
// https://en.cppreference.com/w/cpp/utility/tuple/tie
#include <tuple>

using namespace std;
int main(int argc, char **argv) {

  string filename = argv[1];
  string search_type = argv[2];

  vector<int> labels;
  vector<vector<double>> features;

  tie(labels, features) = read_file(filename);

  auto cache = cache_diff_squared(features);

  set<int> x = {6, 3, 0};

  double y = LOO_accuracy(cache, x, labels);
  cout << y << endl;
  forward(cache, labels);

  return 0;
}
