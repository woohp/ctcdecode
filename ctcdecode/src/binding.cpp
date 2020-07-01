#include "ctc_beam_search_decoder.h"
#include "output.h"
#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>

using namespace std;
namespace py = pybind11;

vector<vector<pair<vector<int>, float>>> beam_decode(
    py::array_t<float> th_probs,
    py::array_t<int> th_seq_lens,
    int beam_size,
    size_t num_processes,
    double cutoff_prob,
    size_t cutoff_top_n,
    size_t blank_id,
    bool log_input)
{
    const int64_t batch_size = th_probs.shape(0);
    const int64_t max_time = th_probs.shape(1);
    const int64_t num_classes = th_probs.shape(2);

    vector<vector<vector<double>>> inputs;
    auto prob_a = th_probs.unchecked<3>();
    auto seq_len_a = th_seq_lens.unchecked<1>();

    for (int b = 0; b < batch_size; ++b)
    {
        // avoid a crash by ensuring that an erroneous seq_len doesn't have us try to access memory we shouldn't
        int seq_len = std::min<int>(seq_len_a[b], max_time);
        vector<vector<double>> temp(seq_len, vector<double>(num_classes));
        for (int t = 0; t < seq_len; ++t)
        {
            for (int n = 0; n < num_classes; ++n)
            {
                float val = prob_a(b,t,n);
                temp[t][n] = val;
            }
        }
        inputs.push_back(temp);
    }

    vector<vector<pair<double, Output>>> batch_results = ctc_beam_search_decoder_batch(
        inputs, beam_size, num_processes, cutoff_prob, cutoff_top_n, blank_id, log_input);

    vector<vector<pair<vector<int>, float>>> output;
    output.reserve(batch_size);

    for (auto& results : batch_results)
    {
        vector<pair<vector<int>, float>> batch_output;
        batch_output.reserve(results.size());  // beam-size

        for (auto& result : results)
        {
            auto& score = result.first;
            auto& output = result.second;
            batch_output.emplace_back(move(output.tokens), score);
        }

        output.push_back(move(batch_output));
    }

    return output;
}

PYBIND11_MODULE(EXTENSION_NAME, m)
{
    m.def("beam_decode", &beam_decode, "beam_decode");
}
