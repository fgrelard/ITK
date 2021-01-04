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


im = sitk.ReadImage("/mnt/d/new_images/Registration/250/Grain3_Xyl/variational/divergence.tif")
im = im[:,:,10]
r=7

X = sitk.GetArrayFromImage(im)
shape = X.shape
X = np.pad(X, (r,r), 'constant')
M = np.zeros_like(X)

diff = 0
w = gaussian_weights(r+diff)
if diff > 0:
    w=w[diff:-diff-1, diff:-diff-1]

edges_h = sobel_h(X)
edges_v = sobel_v(X)
for index in np.ndindex(shape):
    values = X[index[0]-r:index[0]+r+1, index[1]-r:index[1]+r+1].copy()
    # if values.shape[0] > 0 and values.shape[1] > 0:
    #     values[r+1, r+1] = 0
    if (values < 0).any() and (values > 0).any():
        min_d = min([i for i in range(1,r+1)  if (values[r-i:r+i+1,r-i:r+i+1] < 0).any() and (values[r-i:r+i+1,r-i:r+i+1] > 0).any()])
        min_d = find_minimum_radius(values, r)
        vals = values[r-min_d:r+min_d+1, r-min_d:r+min_d+1]
        M[index] = (min_d)# np.median(np.abs(values)) + np.exp(-(min_d-1)**2/1.0)
        if min_d > 1:
            M[index] = M[index] if M[index] > 0 else -M[index]
    if edges_h[index] > 0 or edges_v[index] > 0:
        M[index] = max(edges_h[index], edges_v[index])
    else:
        M[index] = 0

M = local_oriented_max(X, shape)


sigma = np.max(M)/2
# M = np.exp(-M**2/(2*sigma**2))

M=M[r:-r, r:-r]
X=X[r:-r, r:-r]
imO = sitk.ReadImage("/mnt/d/new_images/Registration/250/Grain3_Xyl/variational/fixed.tif")
O = sitk.GetArrayFromImage(imO)
O = sitk.GetArrayFromImage(imO[:,:,0])

mask = np.zeros(M.shape, dtype=bool)
coords = np.argwhere(M > 0)
mask[tuple(coords.T)] = True
markers, _ = ndi.label(mask)

labels = watershed(-M, markers, mask=M)

fig,ax=plt.subplots(2,2)
ax[0,0].imshow(np.abs(X))
ax[0,1].imshow(M)
ax[1,0].imshow(labels)
ax[1,1].imshow(O, cmap="viridis")
ax[1,1].imshow(M,alpha=0.1, cmap="jet")
plt.show()
