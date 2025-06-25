# Spotlight-Resonance-Method
Code implementation of the Spotlight Resonance Method. This is implemented in Numpy, but can be easily adapted to Torch, Tensorflow or alternative.

- SpotlightResonanceMethod.py contains a function for computing spotlight resonance on a given array.

- Example.ipynb contains an example of how to use the Spotlight Resonance Method using a randomly generate numpy array with various anisotropies introduced.


Since the original publication, a futher method has been developed "Privileged Plane Projective Method" (PPPM) which is similar to SRM in function, but provides the actual activations within the plane rather than an angular density. This method may be preferable if complicated distributions of activations make the SRM method difficult to interpret --- in which case PPPM can be used by the developer to interpret the activation distributions by eye.
[A bug with how non-standard bases were displayed should be now corrected]

The following two files contain the code and an example:

- PrivilegedPlaneProjectiveMethod.py contains a function for computing PPPM on a given array.

- Example.ipynb contains an example of how to use the PPPM using a randomly generated numpy array with various anisotropies introduced.
