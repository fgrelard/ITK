import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage.segmentation import watershed
from scipy import ndimage as ndi

from skimage.filters import sobel_h, sobel_v

def gaussian_weights(radius):
    size = 2*radius+1
    sigma = size/4
    return np.array([[np.exp((-i**2-j**2)/(2*sigma**2)) for i in range(-radius,radius+1)] for j in range(-radius,radius+1)])

def find_minimum_radius(values, radius):
    for i in range(1, radius+1):
        new_values_r8 = values[r-i:r+i+1,r-i:r+i+1]
        # new_values_r4 = new_values_r8.copy()
        # new_values_r4[0, 0] = 0
        # new_values_r4[i*2, i*2] = 0
        # new_values_r4[0, i*2] = 0
        # new_values_r4[i*2, 0] = 0
        # if (new_values_r4 < 0).any() and (new_values_r4 > 0).any():
        #     return i
        if (new_values_r8 < 0).any() and (new_values_r8 > 0).any():
            return i# np.sqrt(2*i**2)
    return radius

def local_oriented_max(X, shape):
    r=1
    M = np.zeros_like(X)
    for index in np.ndindex(shape):
        values = np.abs(X[index[0]-r:index[0]+r+1, index[1]-r:index[1]+r+1].copy())
        if (values < 0).any() or (values > 0).any():
            center_value = X[index]
            values_h_0 = values[0:r+1, 0]
            values_h_1 = values[0:r+1, 1]
            values_h_2 = values[0:r+1, 2]

            values_v_0 = values[0, 0:r+1]
            values_v_1 = values[1, 0:r+1]
            values_v_2 = values[2, 0:r+1]

            h_condition = (center_value > values_h_0.max() and center_value > values_h_2.max())
            v_condition = (center_value > values_v_0.max() and center_value > values_v_2.max())
            M[index] = (1 if h_condition or v_condition else 0)
    return M

z = 1
im = sitk.ReadImage("/mnt/d/new_images/Registration/250/Grain3_Xyl/variational/divergence.tif")
im = im[:,:,10]
r=1

X = sitk.GetArrayFromImage(im)

M = X.copy()
sigma = np.max(M)/2
M = 2*(M-1)**2+1
# M = np.exp(-M**2/(2*sigma**2))

imO = sitk.ReadImage("/mnt/d/new_images/Registration/250/Grain3_Xyl/variational/moving.tif")
O = sitk.GetArrayFromImage(imO)
O = sitk.GetArrayFromImage(imO[:,:,z])

mask = np.zeros(M.shape, dtype=bool)
coords = np.argwhere(M > 0)
mask[tuple(coords.T)] = True
markers, _ = ndi.label(mask)

fig,ax=plt.subplots(1,3)
ax[0].imshow(X)
ax[1].imshow(M)
ax[2].imshow(O, cmap="viridis")
ax[2].imshow(M,alpha=0.1, cmap="jet")
plt.show()
