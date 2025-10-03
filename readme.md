I'm trying to implement HRM with cifar-10 dataset

For the initial implementation, I have used a simpler version of the HRM where:
- I avoid using the one step gradient method, and just use normal backpropagation (the time difference seems negligible during the tests, and the the loss drop is more stable)
- The halting mechanism is simplified to a fixed number of steps, instead of a learned halting mechanism.