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

        // print a summary of the dataset
        std::stringstream output;
        output << "Dataset loaded successfully. Total entries: " << index << "\n"
               << "Unique characters: " << this->sorted_states.size() << "\n"
               << "Skipped entries: " << skipped_entries << std::endl;

        spdlog::info(output);

        // -----------------------INITILIZE THE PARAMETERS-----------------------
        // TODO: Done in the main function by calling the initialize_trainsition_probabilities and
        // initialize_emission_probabilities

        // --------------CALCULATE THE ALPHA, BETA AND GAMMA PROBABILITIES--------
        // Then calculate the alpha, beta and gamma probabilities for the current dataset using the
        // initilized parameters
        this->calculate_alpha_beta_gamma();

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
                distribution[i][i] = probability;
                normalization_sum += probability;
            }

            // Another for loop to normalize the probabilities
            for (int j = 0; j < size; ++j)
            {
                distribution[i][j] /= normalization_sum;
            }
        }
    }
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

    // if
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
                distribution[i][i] = probability;
                normalization_sum += probability;
            }

            // Another for loop to normalize the probabilities
            for (int j = 0; j < num_states; ++j)
            {
                distribution[i][j] /= normalization_sum;
            }
        }
    }
}

template <typename T> void HMM<T>::train(int num_itterations)
{
    for (int i = 0; i < num_itterations; i++)
    {
        // TODO: Add training logic here
        // 1. Calculate the forward and backward probabilities using the current parameters
        // 2. Update the transition and emission probabilities using the calculated
        // probabilities
        // 3. Repeat for num_itterations
    }
}
