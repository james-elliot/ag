# A genetic algorithm written in Rust

Parameters for the algorithm are read from a ```<name>.json``` file where ```<name>``` is the name of the program being run. The program looks for such a file in ```./``` and in ```examples/``` in that order.
Parameters can also be specified directly when calling the AG by using the ```Params``` structure. To understand how parameters work, read the commentaries in the ```examples/griewank.json``` file.

The ```examples``` directory contains ```griewank.rs``` which is the code for optimizing the griewank function in dimension 100 along with a proper parameters file.

The griewank example can be run with the command:
>cargo run --example griewank --release

Clustering is rather smart, which means unfortunately that it can be really slow: most of the code is in O(n) while clustering is between O(n^2) and O(n log(n)). In fact, when the population is large, it is by far the slowest part of the program. Dendrograms are slower than dynamic clustering. However clustering is a terrific asset with functions that have a a very large number of local minima. So it is a trade-off...

Parallelism is implemented using Rayon and is only used for the evaluation of population elements. This means mainly two things:
- It is not efficient, so use it only when your evaluation function is ***really*** slow. Using it improperly ***will*** slow down your code!
- As we want to use reference counts to the generic data type inside the chromosome structure to prevent unnecessary copies, we have to use atomic reference counts in order to use parallelism, and this increases a little bit more the cost of using (A)Rc\<T\> instead of the plain \<T\>. Thus some functions (such as crossover and mutation) are slower. You can't have everything... However some others (such as reproduce) are faster, especially when type \<T\> is large. 
