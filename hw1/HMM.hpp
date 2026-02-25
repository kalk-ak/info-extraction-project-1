#include <filesystem>

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

    // TODO: Create a helper function to load the data in memory
    // or initilize a generator for it

    // called from train function to update the trainsitions
    void update_transition();

    // called from the
};
