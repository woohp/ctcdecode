#pragma once

#include <unordered_map>
#include <utility>
#include <vector>

#include "fst/log.h"
#include "output.h"
#include "path_trie.h"

const float NUM_FLT_INF = std::numeric_limits<float>::max();

// inline function for validation check
inline void check(bool x, const char* expr, const char* file, int line, const char* err)
{
    if (!x)
    {
        std::cout << "[" << file << ":" << line << "] ";
        LOG(FATAL) << "\"" << expr << "\" check failed. " << err;
    }
}

#define VALID_CHECK(x, info) check(static_cast<bool>(x), #x, __FILE__, __LINE__, info)
#define VALID_CHECK_EQ(x, y, info) VALID_CHECK((x) == (y), info)
#define VALID_CHECK_GT(x, y, info) VALID_CHECK((x) > (y), info)
#define VALID_CHECK_LT(x, y, info) VALID_CHECK((x) < (y), info)

// Return the sum of two probabilities in log scale
template <typename T>
T log_sum_exp(const T& x, const T& y)
{
    static T num_min = -std::numeric_limits<T>::max();
    if (x <= num_min)
        return y;
    if (y <= num_min)
        return x;
    T xmax = std::max(x, y);
    return std::log(std::exp(x - xmax) + std::exp(y - xmax)) + xmax;
}

// Get pruned probability vector for each time step's beam search
std::vector<std::pair<size_t, float>>
get_pruned_log_probs(const std::vector<float>& prob_step, float cutoff_prob, size_t cutoff_top_n);

// Get beam search result from prefixes in trie tree
std::vector<Output> get_beam_search_result(const std::vector<PathTrie*>& prefixes, size_t beam_size);

// Functor for prefix comparison
bool prefix_compare(const PathTrie* x, const PathTrie* y);

bool prefix_compare_external_scores(
    const PathTrie* x, const PathTrie* y, const std::unordered_map<const PathTrie*, float>& scores);
