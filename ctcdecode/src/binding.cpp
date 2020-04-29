#include "ctc_beam_search_decoder.h"
#include "output.h"
#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <torch/torch.h>
#include <vector>

using namespace std;

vector<at::Tensor> beam_decode(
    at::Tensor th_probs,
    at::Tensor th_seq_lens,
    int beam_size,
    size_t num_processes,
    double cutoff_prob,
    size_t cutoff_top_n,
    size_t blank_id,
    bool log_input)
{
    const int64_t batch_size = th_probs.size(0);
    const int64_t max_time = th_probs.size(1);
    const int64_t num_classes = th_probs.size(2);

    vector<vector<vector<double>>> inputs;
    auto prob_a = th_probs.accessor<float, 3>();
    auto seq_len_a = th_seq_lens.accessor<int, 1>();

    for (int b = 0; b < batch_size; ++b)
    {
        // avoid a crash by ensuring that an erroneous seq_len doesn't have us try to access memory we shouldn't
        int seq_len = std::min<int>(seq_len_a[b], max_time);
        vector<vector<double>> temp(seq_len, vector<double>(num_classes));
        for (int t = 0; t < seq_len; ++t)
        {
            for (int n = 0; n < num_classes; ++n)
            {
                float val = prob_a[b][t][n];
                temp[t][n] = val;
            }
        }
        inputs.push_back(temp);
    }

    vector<vector<pair<double, Output>>> batch_results = ctc_beam_search_decoder_batch(
        inputs, beam_size, num_processes, cutoff_prob, cutoff_top_n, blank_id, log_input);

    auto output = at::empty({ batch_size, beam_size, max_time }, at::kInt);
    // auto timesteps = at::empty({ batch_size, beam_size, max_time }, at::kInt);
    auto scores = at::empty({ batch_size, beam_size }, at::kFloat);
    auto out_length = at::zeros({ batch_size, beam_size }, at::kInt);

    auto outputs_a = output.accessor<int, 3>();
    // auto timesteps_a = timesteps.accessor<int, 3>();
    auto scores_a = scores.accessor<float, 2>();
    auto out_length_a = out_length.accessor<int, 2>();

    for (size_t b = 0; b < batch_results.size(); ++b)
    {
        const auto& results = batch_results[b];
        for (size_t p = 0; p < results.size(); ++p)
        {
            auto& [score, output] = results[p];
            for (size_t t = 0; t < output.tokens.size(); ++t)
            {
                outputs_a[b][p][t] = output.tokens[t];  // fill output tokens
                // timesteps_a[b][p][t] = output.timesteps[t];
            }
            scores_a[b][p] = score;
            out_length_a[b][p] = output.tokens.size();
        }
    }

    return { output, scores, out_length };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("beam_decode", &beam_decode, "beam_decode");
}
