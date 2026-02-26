#include "HMM.hpp"
#include <algorithm>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <spdlog/spdlog.h>
#include <sstream>
#include <unordered_set>
#include <vector>

template <typename T>
HMM<T>::HMM(std::filesystem::path training_dataset) : training_dataset(training_dataset)
{
    // -----------------------LOAD THE DATASET-----------------------
    // populate the data vector with the contents of the training dataset
    std::ifstream file(training_dataset);

    // Log the path of the training dataset being read
    spdlog::info("Reading files from: ", training_dataset.string());

    // check if file exists and is open
    if (not file)
    {
        spdlog::critical("Error opening file: ", training_dataset.string());
        throw std::runtime_error("Failed to open training dataset.");
    }

    // variable to store what is read from the file
    T token;
    long long index = 0; // assuming that the dataset is not bigger than 2^63 characters, which is a
                         // reasonable assumption for this assignment

    long long skipped_entries =
        0; // to keep track of the number of entries that are skipped due to errors

    // have a set to check for unique characters in the dataset and use it to determine the number
    // of states and observations in the HMM
    // ISSUE: one possible issue with this approach is that if a data has never appeared in the
    // training dataset then there would be no state for it according to this initialization.
    // The transition and emission probabilities for that state would be zero but since we are not
    // doing smoothing for the Homework, this is the same as leaving the char from our HMM state and
    // assign it a probabilities of zero if it showes up in the test data and not in training

    while (file >> token)
    {

        if (token.size() != 1)
        {
            std::cerr << "Error: Expected single character, got: " << token << " at index" << index
                      << std::endl;

            continue; // Skip this entry and continue with the next one
        }

        // index to keep track of the number of data read
        index += 1;

        // populate the dataset
        this->data.push_back(token[0]);

        // check if the character has been seen before, if not add it to the state_to_index map and
        // assign it a unique index
        if (this->states.find(token[0]) == this->states.end())
        {
            // add it and assign a new index
            this->states.insert(token);
        }

        // Store all the sets in a vector that is sorted
        this->sorted_states = std::vector<T>(this->states.begin(), this->states.end());
        std::sort(this->sorted_states.begin(), this->sorted_states.end());

        // Create a hashmap to store the mapping from state to index for easy access in the train
        // function
        int length = this->sorted_states.size();
        for (size_t i = 0; i < length; i++)
        {
            this->states_to_index[this->sorted_states[i]] = i;
        }

        // print a summary of the dataset
        std::stringstream output;
        output << "Dataset loaded successfully. Total entries: " << index << "\n"
               << "Unique characters: " << this->sorted_states.size() << "\n"
               << "Skipped entries: " << skipped_entries << std::endl;

        spdlog::info(output);

        // set the flag to true to indicate that the constructor has been called and the parameters
        // have been initialized
        this->constructor_initialized = true;
    };
}

template <typename T>
void HMM<T>::initialize_emission_probabilities(int num_states, bool randomly,
                                               const std::vector<std::vector<double>> &weights)
{

    if (not this->constructor_initialized)
    {
        spdlog::error("initialize_trainsition_probabilities called with out HMM being initialized");
        throw std::runtime_error(
            "initialize_trainsition_probabilities called with out HMM being initialized");
    }

    // NOTE: Here weights is a matrix of size num_states x num_observations where weights[i][j] is
    // the weight for the

    // assert that the weights are provided if randomly is false
    if (not randomly and weights.empty())
    {
        spdlog::error(
            "Must provide weights for the emission probabilities or set randomly flag to true");
        throw std::invalid_argument(
            "Must provide weights for the emission probabilities or set randomly flag to true");
    }

    // if
    if (not randomly)
    {
        this->emission_probabilities = weights;
        //
        // Simple just copy the values from weights (=) here is copy constructor for the vector of
        // vectors
    }
    else
    {
        // variable to modify the original emission probabilities
        std::vector<std::vector<double>> &distribution = this->emission_probabilities;

        // A hardware entropy source (random_device) seeds the Mersenne Twister engine
        std::random_device rd;
        std::mt19937 gen(rd());

        // We want numbers slightly perturbed around a uniform baseline,
        // so we can pull from a uniform real distribution between 0.1 and 1.0
        std::uniform_real_distribution<double> dis(0.01, 1.0);

        // Generate random weights and calculate the total sum
        int size = this->states.size(); // number of unique characters in the dataset, which is the
                                        // number of states and observations in the HMM
        for (int i = 0; i < num_states; ++i)
        {
            double normalization_sum = 0.0; // to normalize the probabilities
            for (int j = 0; j < size; ++j)
            {
                double probability = dis(gen);
                distribution[i][j] = probability;
                normalization_sum += probability;
            }

            // Another for loop to normalize the probabilities
            for (int j = 0; j < size; ++j)
            {
                distribution[i][j] /= normalization_sum;
            }
        }
    }

    this->emission_initialized = true;
}

template <typename T>
void HMM<T>::initialize_trainsition_probabilities(int num_states, bool randomly,
                                                  const std::vector<std::vector<double>> &weights)
{

    if (not this->constructor_initialized)
    {
        spdlog::error("initialize_trainsition_probabilities called with out HMM being initialized");
        throw std::runtime_error(
            "initialize_trainsition_probabilities called with out HMM being initialized");
    }

    // NOTE: Here weights is a matrix of size num_states x num_observations where weights[i][j] is
    // the weight for the

    // assert that the weights are provided if randomly is false
    if (not randomly and weights.empty())
    {
        spdlog::error(
            "Must provide weights for the transition probabilities or set randomly flag to true");
        throw std::invalid_argument(
            "Must provide weights for the transition probabilities or set randomly flag to true");
    }

    if (not randomly)
    {
        this->trainsition_probabilities = weights;
        //
        // Simple just copy the values from weights (=) here is copy constructor for the vector of
        // vectors
    }
    else
    {
        // variable to modify the original transition probabilities
        std::vector<std::vector<double>> &distribution = this->trainsition_probabilities;

        // A hardware entropy source (random_device) seeds the Mersenne Twister engine
        std::random_device rd;
        std::mt19937 gen(rd());

        // We want numbers slightly perturbed around a uniform baseline,
        // so we can pull from a uniform real distribution between 0.1 and 1.0
        std::uniform_real_distribution<double> dis(0.01, 1.0);

        // Generate random weights and calculate the total sum
        for (int i = 0; i < num_states; ++i)
        {
            double normalization_sum = 0.0; // to normalize the probabilities
            for (int j = 0; j < num_states; ++j)
            {
                double probability = dis(gen);
                distribution[i][j] = probability;
                normalization_sum += probability;
            }

            // Another for loop to normalize the probabilities
            for (int j = 0; j < num_states; ++j)
            {
                distribution[i][j] /= normalization_sum;
            }
        }
    }

    this->transition_initialized = true;
}

template <typename T>
void HMM<T>::initialize_initial_state_probabilities(bool randomly,
                                                    const std::vector<double> &weights)
{

    if (not this->constructor_initialized)
    {
        spdlog::error("initialize_trainsition_probabilities called with out HMM being initialized");
        throw std::runtime_error(
            "initialize_trainsition_probabilities called with out HMM being initialized");
    }

    if (not this->emission_initialized)
    {
        spdlog::error("initialize_trainsition_probabilities called with out emission probabilities "
                      "being initialized");
        throw std::runtime_error("initialize_trainsition_probabilities called with out emission "
                                 "probabilities being initialized");
    }

    // NOTE: Here weights is a matrix of size num_states x num_observations where weights[i][j] is
    // the weight for the

    // assert that the weights are provided if randomly is false
    if (not randomly and weights.empty())
    {
        spdlog::error("Must provide weights for the initial state probabilities or set randomly "
                      "flag to true");
        throw std::invalid_argument("Must provide weights for the initial state probabilities or "
                                    "set randomly flag to true");
    }

    if (not randomly)
    {
        this->initial_state_probabilities = weights;
        // Simple just copy the values from weights (=) here is copy constructor for the vector of
        // vectors
    }
    else
    {

        int num_states =
            this->trainsition_probabilities.size(); // Use the actual number of HMM states

        // variable to modify the original transition probabilities
        std::vector<double> &distribution = this->initial_state_probabilities;

        // A hardware entropy source (random_device) seeds the Mersenne Twister engine
        std::random_device rd;
        std::mt19937 gen(rd());

        // assign equal probabilities for all states
        double prob = 1.0 / num_states;
        this->initial_state_probabilities.assign(num_states, prob);
    }

    this->initial_state_probabilities_initialized = true;
}

template <typename T> void HMM<T>::train(int num_itterations)
{
    // -------------------- CHECK IF EVERYTING IS INITIALIZED--------------------
    if (not this->constructor_initialized)
    {
        spdlog::error("train called with out HMM being initialized");
        throw std::runtime_error("train called with out HMM being initialized");
    }

    if (not this->transition_initialized)
    {
        spdlog::error("train called with out transition probabilities being initialized");
        throw std::runtime_error(
            "train called with out transition probabilities being initialized");
    }

    if (not this->emission_initialized)
    {
        spdlog::error("train called with out emission probabilities being initialized");
        throw std::runtime_error("train called with out emission probabilities being initialized");
    }

    if (not this->initial_state_probabilities_initialized)
    {
        spdlog::error("train called with out initial state probabilities being initialized");
        throw std::runtime_error(
            "train called with out initial state probabilities being initialized");
    }

    // cahce the number of states and the number of steps in the dataset for easy access
    const size_t num_states =
        this->trainsition_probabilities.size(); // Use the actual number of HMM states
    const size_t len_data = this->data.size();

    // initialize the alpha, beta, and gamma vectors
    this->alpha.assign(num_states, std::vector<double>(len_data, 0.0));
    this->beta.assign(num_states, std::vector<double>(len_data, 0.0));
    this->gamma.assign(num_states, std::vector<double>(len_data, 0.0));

    // Initialize alpha at t=0 in LOG SPACE
    int first_char_index = this->states_to_index[this->data[0]];
    for (size_t state = 0; state < num_states; state++)
    {
        // log(initial_prob * emission_prob) -> log(initial_prob) + log(emission_prob)
        this->alpha[state][0] = std::log(this->initial_state_probabilities[state]) +
                                std::log(this->emission_probabilities[state][first_char_index]);
    }

    // Initialize beta at the final time step in LOG SPACE
    for (size_t state = 0; state < num_states; state++)
        this->beta[state][len_data - 1] = 0.0; // log(1.0) = 0.0

    // Run Baum-Welch
    for (int iter = 0; iter < num_itterations; iter++)
    {
        this->calculate_alpha_beta_gamma();
        // PERF: The code was written in C++ to make this part of the code fast
        // TODO: Make sure to implement and uncomment update_transition adn update_emission
        // this->update_transition();
        // this->update_emission();
    }
}
//
// Helper function to compute log(a + b) given log(a) and log(b) to avoid underflow issues
double log_sum_exp(double a, double b)
{
    if (a == -std::numeric_limits<double>::infinity())
        return b;
    else if (b == -std::numeric_limits<double>::infinity())
        return a;
    else if (a > b)
        return a + std::log(1.0 + std::exp(b - a));
    else
        return b + std::log(1.0 + std::exp(a - b));
}

// PERF: Optimized to run as fast as possible
template <typename T> void HMM<T>::calculate_alpha_beta_gamma()
{
    int length_of_data = this->data.size();
    int length_of_states = this->trainsition_probabilities.size();

    // Calculate the forward probabilities (alpha)
    for (int i = 1; i < length_of_data; i++)
    {
        // Cache the current data and its index
        T current_observation = this->data[i];
        int current_observation_index = this->states_to_index[current_observation];

        for (int j = 0; j < length_of_states; j++)
        {
            double log_sum = -std::numeric_limits<double>::infinity(); // Represents log(0)

            for (int k = 0; k < length_of_states; k++)
            {
                // alpha[k][i-1] * transition[k][j] := log_alpha + log_transition
                double current_val =
                    alpha[k][i - 1] + std::log(this->trainsition_probabilities[k][j]);

                log_sum = log_sum_exp(log_sum, current_val);
            }

            // Multiply by emission prob := add log_emission
            alpha[j][i] =
                log_sum + std::log(this->emission_probabilities[j][current_observation_index]);
        }
    }

    // Calculate the backward probabilities (beta) next
    for (int t = length_of_data - 2; t >= 0; t--)
    {
        T next_observation = this->data[t + 1];
        int next_obs_index = this->states_to_index[next_observation];

        for (int i = 0; i < length_of_states; i++) // length_of_states is your hidden states
        {
            double log_sum = -std::numeric_limits<double>::infinity();

            for (int j = 0; j < length_of_states; j++)
            {
                // log(transition[i][j]) + log(emission[j][next_obs]) + beta[j][t+1]
                double current_val = std::log(this->trainsition_probabilities[i][j]) +
                                     std::log(this->emission_probabilities[j][next_obs_index]) +
                                     beta[j][t + 1];

                log_sum = log_sum_exp(log_sum, current_val);
            }
            beta[i][t] = log_sum;
        }
    }

    // Calculate the state probabilities (gamma) next
    for (int i = 0; i < length_of_data; i++)
    {
        for (int j = 0; j < length_of_states; j++)
        {
            // gamma[j][i] = alpha[j][i] * beta[j][i] := log_alpha + log_beta
            gamma[j][i] = alpha[j][i] + beta[j][i];
        }
    }
}

template <typename T> void HMM<T>::update_transition()
{
    int numSteps = this->data.size();
    int numStates = this->trainsition_probabilities.size();

    for (int i = 0; i < numStates; i++)
    {
        double log_denominator = -std::numeric_limits<double>::infinity();
        std::vector<double> log_numerators(numStates, -std::numeric_limits<double>::infinity());

        for (int j = 0; j < numStates; j++)
        {
            double log_num_ij = -std::numeric_limits<double>::infinity();

            // Loop through time steps 1 to T-1
            for (int t = 1; t < numSteps; t++)
            {
                int obs_index = this->states_to_index[this->data[t]];

                // xi_t(i,j) = alpha[i][t-1] * trans[i][j] * emit[j][O_t] * beta[j][t]
                double current_xi =
                    this->alpha[i][t - 1] + std::log(this->trainsition_probabilities[i][j]) +
                    std::log(this->emission_probabilities[j][obs_index]) + this->beta[j][t];

                log_num_ij = log_sum_exp(log_num_ij, current_xi);
            }
            log_numerators[j] = log_num_ij;
            log_denominator = log_sum_exp(log_denominator, log_num_ij);
        }

        // Update the actual transition matrix
        for (int j = 0; j < numStates; j++)
        {
            // Division in log space is subtraction
            double log_prob = log_numerators[j] - log_denominator;

            // Convert back to normal space
            this->trainsition_probabilities[i][j] = std::exp(log_prob);
        }
    }
}

template <typename T> void HMM<T>::update_emission()
{
    int numSteps = this->data.size();
    int numStates = this->trainsition_probabilities.size();
    int numSymbols = this->sorted_states.size();

    for (int j = 0; j < numStates; j++)
    {
        double log_denominator = -std::numeric_limits<double>::infinity();
        std::vector<double> log_numerators(numSymbols, -std::numeric_limits<double>::infinity());

        for (int t = 0; t < numSteps; t++)
        {
            // gamma_t(j) = alpha_t(j) * beta_t(j)
            // Note: We can just use the gamma matrix you already calculated!
            double current_gamma = this->gamma[j][t];

            // Add to the total expected time spent in state j (Denominator)
            log_denominator = log_sum_exp(log_denominator, current_gamma);

            // Add to the expected time spent in state j emitting specific character (Numerator)
            int obs_index = this->states_to_index[this->data[t]];
            log_numerators[obs_index] = log_sum_exp(log_numerators[obs_index], current_gamma);
        }

        // Update the actual emission matrix
        for (int k = 0; k < numSymbols; k++)
        {
            // Division in log space is subtraction
            double log_prob = log_numerators[k] - log_denominator;

            // Convert back to normal space
            this->emission_probabilities[j][k] = std::exp(log_prob);
        }
    }
}
