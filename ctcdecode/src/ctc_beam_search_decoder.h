#pragma once

#include <string>
#include <utility>
#include <vector>

#include "output.h"
#include "path_trie.h"

/* CTC Beam Search Decoder

 * Parameters:
 *     probs_seq: 2-D vector that each element is a vector of probabilities
 *               over vocabulary of one time step.
 *     beam_size: The width of beam search.
 *     cutoff_prob: Cutoff probability for pruning.
 *     cutoff_top_n: Cutoff number for pruning.
 * Return:
 *     A vector that each element is a pair of score  and decoding result,
 *     in desending order.
*/

std::vector<Output> ctc_beam_search_decoder(
    const std::vector<std::vector<float>>& probs_seq,
    int beam_size,
    float cutoff_prob = 1.0,
    size_t cutoff_top_n = 40,
    size_t blank_id = 0);

/* CTC Beam Search Decoder for batch data

 * Parameters:
 *     probs_seq: 3-D vector that each element is a 2-D vector that can be used
 *                by ctc_beam_search_decoder().
 *     beam_size: The width of beam search.
 *     num_processes: Number of threads for beam search.
 *     cutoff_prob: Cutoff probability for pruning.
 *     cutoff_top_n: Cutoff number for pruning.
 * Return:
 *     A 2-D vector that each element is a vector of beam search decoding
 *     result for one audio sample.
*/
std::vector<std::vector<Output>> ctc_beam_search_decoder_batch(
    const std::vector<std::vector<std::vector<float>>>& probs_split,
    int beam_size,
    size_t num_processes,
    float cutoff_prob = 1.0,
    size_t cutoff_top_n = 40,
    size_t blank_id = 0);

class DecoderState
{
    int abs_time_step;
    size_t beam_size;
    float cutoff_prob;
    size_t cutoff_top_n;
    size_t blank_id;

    std::vector<PathTrie*> prefixes;
    PathTrie root;

public:
    /* Initialize CTC beam search decoder for streaming
     *
     * Parameters:
     *     beam_size: The width of beam search.
     *     cutoff_prob: Cutoff probability for pruning.
     *     cutoff_top_n: Cutoff number for pruning.
     */
    DecoderState(size_t beam_size, float cutoff_prob, size_t cutoff_top_n, size_t blank_id);
    ~DecoderState() = default;

    /* Process logits in decoder stream
     *
     * Parameters:
     *     probs: 2-D vector where each element is a vector of probabilities
     *               over alphabet of one time step.
     */
    void next(const std::vector<std::vector<float>>& probs_seq);

    /* Get current transcription from the decoder stream state
     *
     * Return:
     *     A vector where each element is a pair of score and decoding result,
     *     in descending order.
     */
    std::vector<Output> decode() const;
};
