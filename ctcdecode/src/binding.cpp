#include "ctc_beam_search_decoder.h"
#include "output.h"
#include <algorithm>
#include <iostream>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

using namespace std;
namespace py = pybind11;

vector<vector<pair<vector<int>, float>>> beam_decode(
    py::array_t<float> log_probs,
    py::array_t<int> seq_lens,
    int beam_size,
    size_t num_processes,
    float cutoff_prob,
    size_t cutoff_top_n,
    size_t blank_id)
{
    const int64_t batch_size = log_probs.shape(0);
    const int64_t max_time = log_probs.shape(1);
    const int64_t num_classes = log_probs.shape(2);

    vector<vector<vector<float>>> inputs;
    auto log_prob_a = log_probs.unchecked<3>();
    auto seq_len_a = seq_lens.unchecked<1>();

    for (int b = 0; b < batch_size; ++b)
    {
        // avoid a crash by ensuring that an erroneous seq_len doesn't have us try to access memory we shouldn't
        int seq_len = std::min<int>(seq_len_a[b], max_time);
        vector<vector<float>> temp(seq_len, vector<float>(num_classes));
        for (int t = 0; t < seq_len; ++t)
        {
            for (int n = 0; n < num_classes; ++n)
            {
                temp[t][n] = log_prob_a(b, t, n);
            }
        }
        inputs.push_back(temp);
    }

    vector<vector<Output>> batch_results
        = ctc_beam_search_decoder_batch(inputs, beam_size, num_processes, cutoff_prob, cutoff_top_n, blank_id);

    vector<vector<pair<vector<int>, float>>> output;
    output.reserve(batch_size);

    for (auto& results : batch_results)
    {
        vector<pair<vector<int>, float>> batch_output;
        batch_output.reserve(results.size());  // beam-size

        for (auto& result : results)
            batch_output.emplace_back(move(result.tokens), result.score);

        output.push_back(move(batch_output));
    }

    return output;
}

PYBIND11_MODULE(ctc_decode, m)
{
    using namespace pybind11::literals;

    m.def(
        "beam_decode",
        &beam_decode,
        "beam_decode",
        "log_probs"_a,
        "seq_lens"_a,
        "beam_size"_a,
        "num_processes"_a,
        "cutoff_prob"_a,
        "cutoff_top_n"_a,
        "blank_id"_a);
}
