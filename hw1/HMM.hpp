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
    // HACK: Using modern C++ features like std::vector and std::filesystem, we can rely on the
    // automatic memory
    ~HMM();

    // Used the BAUM welch algorithm to train the HMM for num_itterations
    void train(int num_itterations);

  private:
    std::filesystem::path training_dataset;

    // Pointer to the dataset. Allocated in the constructor and deallocated in the destructor
    // Using the heap so that the code can be used for much bigger datasets than the memory can hold
    // at once. The data can be loaded in batches in the train function
    std::vector<char> data;

    // NOTE: n x n matrix where n is the number of states in the HMM
    std::vector<std::vector<double>>
        trainsition_probabilities; // Matrix representing the transition probabilities between
                                   // states
                                   // NOTE: has size num_states x num_observations

    std::vector<std::vector<double>>
        emission_probabilities; // Matrix representing the emission probabilities of the
                                // observations given the states

    // All these probabilities are initilized when the Constructor is called and updated when
    // the train function is called
    std::vector<double> alpha; // cache for forward probabilities
    std::vector<double> beta;  // cache for backward probabilities
    std::vector<double> gamma; // cache for state probabilities

    // function to calculate the alpha, beta and gamma probabilities for the current dataset. Called
    // from the train function
    void calculate_alpha_beta_gamma();

    // called from train function to update the trainsitions
    void update_transition();

    // Called from the train function to update the emission probabilities
    void update_emission();
};
