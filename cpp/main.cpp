#include "nn.h"
#include "read.h"
#include "search.h"
// https://stackoverflow.com/questions/728068/how-to-calculate-a-time-difference-in-c
#include <string>
// https://en.cppreference.com/w/cpp/utility/tuple/tie
#include <tuple>

using namespace std;
int main(int argc, char **argv) {

  if (argc != 3) {
    cout << "Usage: ./a.out <path_to_text_file> <search_type>" << endl;
    cout << "\tValid search types: [forward, backward, both]" << endl;
    exit(1);
  }

  string filename = argv[1];
  string search_type = argv[2];

  vector<int> labels;
  vector<vector<double>> features;

  tie(labels, features) = read_file(filename);

  auto cache = cache_diff_squared(features);

  if (search_type == "forward") {
    forward(cache, labels);
  } else if (search_type == "backward") {
    backward(cache, labels);
  } else if (search_type == "both") {
    cout << "Running forward:" << endl;
    forward(cache, labels);
    cout << "Running backward:" << endl;
    backward(cache, labels);
  } // forward(cache, labels);

  return 0;
}
