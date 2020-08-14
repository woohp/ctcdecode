#include "ctc_beam_search_decoder.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <utility>

#include "decoder_utils.h"
#include "fst/fstlib.h"
#include "output.h"
#include "path_trie.h"
#include "thread_pool.h"
using namespace std;

DecoderState::DecoderState(size_t beam_size, double cutoff_prob, size_t cutoff_top_n, size_t blank_id, bool log_input)
    : abs_time_step(0)
    , beam_size(beam_size)
    , cutoff_prob(cutoff_prob)
    , cutoff_top_n(cutoff_top_n)
    , blank_id(blank_id)
    , log_input(log_input)
{
    // init prefixes' root
    root.score = root.log_prob_b_prev = 0.0;
    prefixes.push_back(&root);
}

void DecoderState::next(const vector<vector<double>>& probs_seq)
{
    // dimension check
    size_t num_time_steps = probs_seq.size();

    // prefix search over time
    for (size_t time_step = 0; time_step < num_time_steps; ++time_step, ++abs_time_step)
    {
        auto& prob = probs_seq[time_step];

        float min_cutoff = -NUM_FLT_INF;
        bool full_beam = false;

        vector<pair<size_t, float>> log_prob_idx = get_pruned_log_probs(prob, cutoff_prob, cutoff_top_n, log_input);
        // loop over chars
        for (size_t index = 0; index < log_prob_idx.size(); index++)
        {
            auto c = log_prob_idx[index].first;
            auto log_prob_c = log_prob_idx[index].second;

            for (size_t i = 0; i < prefixes.size() && i < beam_size; ++i)
            {
                auto prefix = prefixes[i];
                if (full_beam && log_prob_c + prefix->score < min_cutoff)
                {
                    break;
                }
                // blank
                if (c == blank_id)
                {
                    prefix->log_prob_b_cur = log_sum_exp(prefix->log_prob_b_cur, log_prob_c + prefix->score);
                    continue;
                }
                // repeated character
                if (c == prefix->character)
                {
                    prefix->log_prob_nb_cur
                        = log_sum_exp(prefix->log_prob_nb_cur, log_prob_c + prefix->log_prob_nb_prev);
                }
                // get new prefix
                auto prefix_new = prefix->get_path_trie(c, abs_time_step, log_prob_c);

                if (prefix_new != nullptr)
                {
                    float log_p = -NUM_FLT_INF;

                    if (c == prefix->character && prefix->log_prob_b_prev > -NUM_FLT_INF)
                    {
                        log_p = log_prob_c + prefix->log_prob_b_prev;
                    }
                    else if (c != prefix->character)
                    {
                        log_p = log_prob_c + prefix->score;
                    }

                    prefix_new->log_prob_nb_cur = log_sum_exp(prefix_new->log_prob_nb_cur, log_p);
                }
            }  // end of loop over prefix
        }  // end of loop over vocabulary

        prefixes.clear();
        // update log probs
        root.iterate_to_vec(prefixes);

        // only preserve top beam_size prefixes
        if (prefixes.size() >= beam_size)
        {
            nth_element(prefixes.begin(), prefixes.begin() + beam_size, prefixes.end(), prefix_compare);
            for (size_t i = beam_size; i < prefixes.size(); ++i)
            {
                prefixes[i]->remove();
            }

            prefixes.resize(beam_size);
        }
    }  // end of loop over time
}

vector<Output> DecoderState::decode() const
{
    vector<PathTrie*> prefixes_copy = prefixes;
    unordered_map<const PathTrie*, float> scores;
    for (PathTrie* prefix : prefixes_copy)
    {
        scores[prefix] = prefix->score;
    }

    using namespace placeholders;
    size_t num_prefixes = min(prefixes_copy.size(), beam_size);
    sort(
        prefixes_copy.begin(),
        prefixes_copy.begin() + num_prefixes,
        bind(prefix_compare_external_scores, _1, _2, scores));

    // compute aproximate ctc score as the return score, without affecting the
    // return order of decoding result. To delete when decoder gets stable.
    for (size_t i = 0; i < beam_size && i < prefixes_copy.size(); ++i)
    {
        double approx_ctc = scores[prefixes_copy[i]];
        prefixes_copy[i]->approx_ctc = approx_ctc;
    }

    return get_beam_search_result(prefixes_copy, beam_size);
}

vector<Output> ctc_beam_search_decoder(
    const vector<vector<double>>& probs_seq,
    int beam_size,
    double cutoff_prob,
    size_t cutoff_top_n,
    size_t blank_id,
    bool log_input)
{
    DecoderState state(beam_size, cutoff_prob, cutoff_top_n, blank_id, log_input);
    state.next(probs_seq);
    return state.decode();
}

vector<vector<Output>> ctc_beam_search_decoder_batch(
    const vector<vector<vector<double>>>& probs_split,
    int beam_size,
    size_t num_processes,
    double cutoff_prob,
    size_t cutoff_top_n,
    size_t blank_id,
    bool log_input)
{
    VALID_CHECK_GT(num_processes, 0, "num_processes must be nonnegative!");
    // thread pool
    thread_pool pool(num_processes);
    // number of samples
    size_t batch_size = probs_split.size();

    // enqueue the tasks of decoding
    vector<vector<Output>> outputs(batch_size);

    pool.parallel_for(0, batch_size, [&](size_t i, size_t) {
        outputs[i] = ctc_beam_search_decoder(probs_split[i], beam_size, cutoff_prob, cutoff_top_n, blank_id, log_input);
    });

    return outputs;
}
