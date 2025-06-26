# Notes on the images in this folder.

There are 6 files so far, all displayed using the Privileged Plane Projective Method. 
[TITLE, AUTOENCODER SIZE, EXTRINSIC DIMENSION OF BOTTLENECK, THOMPSON BASIS VECTORS, EPSILON]
- MNIST Small 16 17 epsilon=0.85
- MNIST Small 16 32 epsilon=0.85
- MNIST Small 16 64 epsilon=0.85
- MNIST Large 16 17 epsilon=0.85
- MNIST Large 16 32 epsilon=0.85
- MNIST Large 16 64 epsilon=0.85

These are then shown as a series of plots, starting with initialisations in the two leftmost columns then progressing the following columns shows how the representations shift throughout training. Each column is an additional +number amount of epochs trained compared to the previous. (bacth size 24 and computed over training set)
[init, init, +2, +2, +2, +2, +2, +2, +2, +2, +2, +2, +5, +5, +10, +10, +10, +10, +10, +10, +10]

I have chosen to display this as a density plot as opposed to a scatter, though either would suffice. This is then displayed using matplotlib imshow. It is also normalised, since the anti-aligned angles can cover more volume they could have greater density. Therefore, I used a random normal to approximate how an *isotropic* **representation** distribution would appear, and divided by this to act as a volume normaliser.

```python
def gaussian(X, Y, mean, spread):
    exponent = -(np.square(X-mean[0])+np.square(Y-mean[1]))/(2*spread*spread)
    return np.exp(exponent)


RESOLUTION = 300
X = np.linspace(-maximum, maximum, RESOLUTION)
Y = np.linspace(-maximum, maximum, RESOLUTION)
X, Y = np.meshgrid(X, Y)

spread = 0.2/maximum

heat_map = np.zeros_like(X)
for i in range(projected_activations.shape[0]):
    heat_map += gaussian(X, Y, projected_activations[i, :], spread)
```