#include "decoder_utils.h"

#include <algorithm>
#include <cmath>
#include <limits>
using namespace std;

vector<pair<size_t, float>>
get_pruned_log_probs(const vector<double>& prob_step, double cutoff_prob, size_t cutoff_top_n, bool log_input)
{
    vector<pair<int, double>> prob_idx;
    prob_idx.reserve(prob_step.size());
    const double log_cutoff_prob = log(cutoff_prob);
    for (size_t i = 0; i < prob_step.size(); ++i)
    {
        prob_idx.push_back(pair<int, double>(i, prob_step[i]));
    }

    // pruning of vacobulary
    size_t cutoff_len = prob_step.size();
    if (log_cutoff_prob < 0.0 || cutoff_top_n < cutoff_len)
    {
        sort(prob_idx.begin(), prob_idx.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
        if (log_cutoff_prob < 0.0)
        {
            double cum_prob = 0.0;
            cutoff_len = 0;
            for (size_t i = 0; i < prob_idx.size(); ++i)
            {
                cum_prob = log_sum_exp(cum_prob, log_input ? prob_idx[i].second : log(prob_idx[i].second));
                cutoff_len += 1;
                if (cum_prob >= cutoff_prob || cutoff_len >= cutoff_top_n)
                    break;
            }
        }
        else
        {
            cutoff_len = cutoff_top_n;
        }
        prob_idx.resize(cutoff_len);
    }

    vector<pair<size_t, float>> log_prob_idx;
    for (size_t i = 0; i < cutoff_len; ++i)
    {
        log_prob_idx.push_back(pair<int, float>(
            prob_idx[i].first,
            log_input ? prob_idx[i].second : log(prob_idx[i].second + numeric_limits<float>::min())));
    }
    return log_prob_idx;
}

vector<Output> get_beam_search_result(const vector<PathTrie*>& prefixes, size_t beam_size)
{
    // allow for the post processing
    vector<PathTrie*> space_prefixes(prefixes.begin(), prefixes.begin() + min(beam_size, prefixes.size()));

    sort(space_prefixes.begin(), space_prefixes.end(), prefix_compare);
    vector<Output> output_vecs;
    output_vecs.reserve(space_prefixes.size());
    for (size_t i = 0; i < beam_size && i < space_prefixes.size(); ++i)
    {
        vector<int> tokens;
        vector<int> timesteps;
        space_prefixes[i]->get_path_vec(tokens, timesteps);
        output_vecs.emplace_back(-space_prefixes[i]->approx_ctc, tokens, timesteps);
    }

    return output_vecs;
}

bool prefix_compare(const PathTrie* x, const PathTrie* y)
{
    if (x->score == y->score)
    {
        if (x->character == y->character)
        {
            return false;
        }
        else
        {
            return (x->character < y->character);
        }
    }
    else
    {
        return x->score > y->score;
    }
}

bool prefix_compare_external_scores(
    const PathTrie* x, const PathTrie* y, const unordered_map<const PathTrie*, float>& scores)
{
    if (scores.at(x) == scores.at(y))
    {
        if (x->character == y->character)
        {
            return false;
        }
        else
        {
            return (x->character < y->character);
        }
    }
    else
    {
        return scores.at(x) > scores.at(y);
    }
}
