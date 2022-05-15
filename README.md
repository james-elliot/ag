# A genetic algorithm written in Rust
Parameters for the algorithm are read from a parameters.json file. The program looks for such a file in ./parameters.json , ~/parameters.json and examples/parameters.json

Parameters can also be specified directly when calling the AG by using the Params structure.

The examples directory contains griewank.rs which is the optimization of the griewank function in dimension 100 along with a parameters.json file.

The griewank example can be run with the command:

cargo run --example griewank --release


