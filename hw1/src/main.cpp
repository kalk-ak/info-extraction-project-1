int main()
{
    // setup input file paths
    fs::path train_file = data_dir / "textA-1.txt";
    fs::path test_file = data_dir / "textB-1.txt";

    // setup output file paths dynamically based on state count
    fs::path model_file = models_dir / ("hmm_model_" + std::to_string(num_states) + "state.txt");
    fs::path csv_file = models_dir / ("metrics_" + std::to_string(num_states) + "state.csv");

    try
    {
        // initialize the HMM with the training data
        HMM<T> hmm(train_file);

        // check if we already have a saved model to skip training
        if (fs::exists(model_file))
        {
            spdlog::info("Found existing {}-state model. Loading...", num_states);
            hmm.load_model(model_file);

            // set pi to uniform probabilities (1/N)
            std::vector<double> pi(num_states, 1.0 / num_states);
            hmm.initialize_initial_state_probabilities(false, pi);

            // run evaluation on the test set
            double test_likelihood = hmm.test(test_file);
            spdlog::info("Final Test Log-Likelihood: {:.4f}", test_likelihood);
        }
        else
        {
            spdlog::info("No model found. Initializing {}-State training...", num_states);

            // specific initialization for the 2-state model from part 1 of the homework
            if (num_states == 2)
            {
                // transition probabilities setup
                std::vector<std::vector<double>> transition_probs = {{0.49, 0.51}, {0.51, 0.49}};
                hmm.initialize_trainsition_probabilities(2, false, transition_probs);

                // emission probabilities setup
                // NOTE: space is at index 0, 'a' is at index 1, 'n' is at index 14
                std::vector<double> state1_emissions(27, 0.0370);
                state1_emissions[0] = 0.0367;

                std::vector<double> state2_emissions(27, 0.0371);
                state2_emissions[0] = 0.0367;
                state2_emissions[14] = 0.0370;

                hmm.initialize_emission_probabilities(2, false,
                                                      {state1_emissions, state2_emissions});
            }
            else
            {
                // for 4-states or other N, use random initialization to break symmetry
                // this is needed for the Baum-Welch algorithm to converge properly
                hmm.initialize_trainsition_probabilities(num_states, true);
                hmm.initialize_emission_probabilities(num_states, true);
            }

            // initial state probabilities (pi) set to uniform 1/N
            std::vector<double> pi(num_states, 1.0 / num_states);
            hmm.initialize_initial_state_probabilities(false, pi);

            // run Baum-Welch training for 600 iterations
            int num_iterations = 600;
            spdlog::info("Training and evaluating for {} iterations...", num_iterations);

            // training handles logging to the csv_file path
            hmm.train(num_iterations, test_file, csv_file.string());

            // save the finished model to the models directory
            hmm.save_model(model_file);
        }
    }
    catch (const std::exception &e)
    {
        spdlog::error("Error in run_hmm: {}", e.what());
        throw;
    }
}

int main(int argc, char *argv[])
{
    // default parameters
    int num_states = 2;
    std::string data_type = "char";

    // check command line for state count argument
    if (argc > 1)
    {
        try
        {
            num_states = std::stoi(argv[1]);
            spdlog::info("State count set to N = {}", num_states);
        }
        catch (...)
        {
            spdlog::warn("Invalid state count argument. Defaulting to 2");
        }
    }

    // check command line for data type argument
    if (argc > 2)
    {
        data_type = argv[2];
        spdlog::info("Data type set to {}", data_type);
    }

    // setup directories
    fs::path data_dir = "data";
    fs::path models_dir = "models";

    // ensure output directory exists
    try
    {
        fs::create_directories(models_dir);

        // choose the correct template instantiation based on command line string
        if (data_type == "char")
        {
            run_hmm<char>(num_states, data_dir, models_dir);
        }
        else if (data_type == "int")
        {
            run_hmm<int>(num_states, data_dir, models_dir);
        }
        else if (data_type == "double")
        {
            run_hmm<double>(num_states, data_dir, models_dir);
        }
        else if (data_type == "float")
        {
            run_hmm<float>(num_states, data_dir, models_dir);
        }
        else if (data_type == "string")
        {
            run_hmm<std::string>(num_states, data_dir, models_dir);
        }
        else
        {
            spdlog::error("Unsupported data type: {}. Use char, int, double, or float.", data_type);
            return 1;
        }
    }
    catch (const std::exception &e)
    {
        spdlog::critical("Fatal error during execution: {}", e.what());
        return 1;
    }

    return 0;
}
