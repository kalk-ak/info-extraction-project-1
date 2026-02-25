cmake_minimum_required(VERSION 3.16)
project(HMM VERSION 1.0 LANGUAGES CXX)

# ----------- DOWNLOAD ALL DEPENDENCIES AUTOMATICALLY ------------------
# Make sure that compile_comands.json is created to avoid LSP warnings
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)


FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.13.0   # use stable version
)


FetchContent_MakeAvailable(spdlog)

# ---------- END of DOWNLOAD ALL DEPENDENCIES AUTOMATICALLY ------------------


# Define the sources for the executable
vars(
	SOURCES
	main.cpp
	HMM.cpp
)
# create the executable
add_executable(
	HMM	
	${SOURCES}
)

# Use C++ 17
target_compile_features(daq_server PRIVATE cxx_std_17)

# All headers are in the include directory, so add it to the include path
target_include_directories(HMM PRIVATE include)

# Link the spdlog library to the executable
target_link_libraries(HMM PRIVATE spdlog::spdlog)
