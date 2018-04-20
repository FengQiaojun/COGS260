Lecture 2 April 5th



Reading:

- Edge Detection

#### A Computational Approach to Edge Detection (PAMI 1986) 

3 criteria for edge detector: 

1. Good detection (high recall(low false negative), high precise(low false positive))
2. Good localization (close to the center of the edge)
3. Only one response to a single edge 

Through spatial scaling of f we can trade off detection performance against localization, but we cannot improve both simultaneously.



#### Supervised Learning of Edges and Object Boundaries (CVPR 2006)





#### Holistically-Nested Edge Detection (ICCV 2015)

address 2 important issues

1. holistic image training and prediction
2. multi-scale and multi-level feature learning

the performance gain is largely due to three aspects: 

1. FCN-like image-to-image training allows us to simultaneouslytrain on a significantly larger amount of samples
2. deep supervision in our model guides the learning of more transparent features
3. interpolating the side outputs in the end-to-end learning encourages coherent contributions from each layer

For different layers with different size output, supervised with different size of groundtruth.

