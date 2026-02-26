#include <filesystem>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

template <typename T> class HMM
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

    void print_parameters() const; // for debugging purposes, prints the transition and emission
                                   // probabilities to the console

    void initialize_initial_state_probabilities(
        bool randomly = false, const std::vector<double> &weights = {}); // initializes the
                                                                         // initial state
                                                                         // probabilities
    void initialize_trainsition_probabilities(
        int num_states, bool randomly = false,
        const std::vector<std::vector<double>> &weights =
            {}); // initializes the transition probabilities randomly or with
                 // a constructor Returns true if successful, false otherwise

    // Called to initialize_trainsition_probabilities in main
    void initialize_emission_probabilities(int num_states, bool randomly = false,
                                           const std::vector<std::vector<double>> &weights = {});

  private:
    std::filesystem::path training_dataset;

    // Pointer to the dataset. Allocated in the constructor and deallocated in the destructor
    // Using the heap so that the code can be used for much bigger datasets than the memory can hold
    // at once. The data can be loaded in batches in the train function
    std::vector<T> data;

    // hash-map to store state to index mapping
    std::unordered_set<T> states; // Maps each state to a unique index for easy access in
                                  // the transition and emission probability matrices

    bool constructor_initialized = false; // flag to check if the constructor has been called and
                                          // the parameters have been initialized
    bool transition_initialized = false;
    bool emission_initialized = false;
    bool initial_state_probabilities_initialized = false;

    // vector to store the unique states in the dataset, index indicates the index of the state in
    // the transition and emission probability matrices
    std::vector<T> sorted_states;

    std::unordered_map<T, int>
        states_to_index; // Maps each state to a unique index for easy access in the transition and
                         // emission probability matrices

    std::vector<double> initial_state_probabilities; // 'pi' in HMM literature

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

    std::vector<std::vector<double>> alpha; // cache for forward probabilities
    std::vector<std::vector<double>> beta;
    std::vector<std::vector<double>> gamma; // cache for state probabilities

    // function to calculate the alpha, beta and gamma probabilities for the current dataset. Called
    // from the train function
    void calculate_alpha_beta_gamma();

    // called from train function to update the trainsitions
    void update_transition();

    // Called from the train function to update the emission probabilities
    void update_emission();
};
