#pragma once
#include <vector>

/* Struct for the beam search output, containing the tokens based on the vocabulary indices, and the timesteps
 * for each token in the beam search output
 */
struct Output
{
    double score;
    std::vector<int> tokens, timesteps;

    Output() = default;

    Output(double score, const std::vector<int>& tokens, const std::vector<int> timesteps)
        : score(score)
        , tokens(tokens)
        , timesteps(timesteps)
    {}
};
