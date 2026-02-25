#include "HMM.hpp"
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <vector>

HMM::HMM(std::filesystem::path training_dataset) : training_dataset(training_dataset)
{
    // -----------------------LOAD THE DATASET-----------------------
    // populate the data vector with the contents of the training dataset
    std::ifstream file(training_dataset);

    if (not file)
        std::cerr << "Error opening file: " << training_dataset << std::endl;

    std::string word;
    long long index = 0; // assuming that the dataset is not bigger than 2^63 characters, which is a
                         // reasonable assumption for this assignment

    long long skipped_entries =
        0; // to keep track of the number of entries that are skipped due to errors

    // have a set to check for unique characters in the dataset and use it to determine the number
    // of states and observations in the HMM
    // ISSUE: one possible issue with this approach is that if a data has never appeared in the
    // training dataset then there would be no state for it according to the baul-welch algorithm,
    // the transition and emission probabilities for that state would be zero but since we are not
    // doing smoothing for the Homework, this is the same as leaving the char from our HMM state and
    // assign it a probabilities of zero if it showes up in the test data
    std::unordered_set<char> unique_characters;

    while (file >> word)
    {

        if (word.size() != 1)
        {
            std::cerr << "Error: Expected single character, got: " << word << " at index" << index
                      << std::endl;

            continue; // Skip this entry and continue with the next one
        }

        // index to keep track of the number of data read
        index += 1;

        // populate the dataset
        this->data.push_back(word[0]);

        // add to the set to keep tracck of unique characters
        unique_characters.insert(word[0]);
    }
    // print a summary of the dataset
    std::cout << "Dataset loaded successfully. Total entries: " << index << "\n"
              << "Unique characters: " << unique_characters.size() << "\n"
              << "Skipped entries: " << skipped_entries << std::endl;

    // -----------------------INITILIZE THE PARAMETERS-----------------------
    // TODO: Done in the main function by calling the initialize_trainsition_probabilities and
    // initialize_emission_probabilities

    // --------------CALCULATE THE ALPHA, BETA AND GAMMA PROBABILITIES--------
    // Then calculate the alpha, beta and gamma probabilities for the current dataset using the
    // initilized parameters
    this->calculate_alpha_beta_gamma();
}

bool HMM::initialize_emission_probabilities(bool randomly,
                                            const std::vector<std::vector<int>> &weights)
{
}

void HMM::train(int num_itterations)
{
    for (int i = 0; i < num_itterations; i++)
    {
        // TODO: Add training logic here
        // 1. Calculate the forward and backward probabilities using the current parameters
        // 2. Update the transition and emission probabilities using the calculated probabilities
        // 3. Repeat for num_itterations
    }
}
