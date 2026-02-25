#pragma once
#include <filesystem>
#include <vector>

class HMM
{
  public:
    // Constructor for the HMM.
    //
    // Responsible for reading the dataset and storing it in memory and initilizing the parameters
    // randomly
    HMM(std::filesystem::path training_dataset);

    // Destructor for the HMM. Responsible for closing the file if needed and releasing the
    // initilized data from memory
    ~HMM();

    // Used the BAUM welch algorithm to train the HMM for num_itterations
    void train(int num_itterations);

  private:
    std::filesystem::path training_dataset;

    // All these probabilities are initilized when the Constructor is called and updated when the
    // train function is called
    std::vector<double> alpha; // cache for forward probabilities
    std::vector<double> beta;  // cache for backward probabilities
    std::vector<double> gamma; // cache for state probabilities

    // TODO: Create a helper function to load the data in memory
    // or initilize a generator for it

    // called from train function to update the trainsitions
    void update_transition();

    // Called from the train function to update the emission probabilities
    void update_emission();
};
