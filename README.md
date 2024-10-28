# Session 2: Image stacking, Photometry and Data analysis
## Hosted and written by Rutvik Hegde



> [!NOTE] Note
> All .fits data is public and has been taken from https://archive.stsci.edu/prepds/hugs/

## Image Stacking
Image stacking and processing are techniques used to transform raw data into clear, detailed images. In the world of optics and electronics, from cameras to telescopes, capturing sharp, noise-free images often requires more than a single snapshot. Image stacking involves combining multiple images of the same scene or object to improve quality, reduce noise, and enhance details that might otherwise be missed. This process is especially valuable in fields like astronomy, where light is faint and exposures are long, and in medical imaging, where clarity is critical.

Various algorithms have been developed to suit different needs—whether it's aligning slightly shifted images, minimizing blur, or enhancing specific features. For instance, in astrophotography, stacking images helps reveal faint stars and galaxies that might not appear in a single shot. This approach, common across many optical devices, ensures that the final image is rich in detail and as accurate as possible to the original scene or subject.

### How does image stacking work?

A camera is basically a sensor that creates an intensity map, and stores it as raw data. If, suppose, a location is bright, it gets a higher value, and if it is dark, it gets a lower value. This intensity map is created for a specific wavelength, say 450nm. There are multiple sensors/filters which each look at different wavelengths, and hence produce different intensity maps for that specific wavelength.

This data is stored in arrays of 1 dimension, i.e a sequence of values indicating the intensity/brightness for each pixel. Combining all of this data results in an intensity map that contains multiple dimensions, with each dimension corresponding to a filter.

_If the sensor took data from 5 filters, you get a 5D array. If it took data from 3 filters you get a 3D array._

Now, conventional displays can only show 3 colours, RGB. Hence they can display 3 dimensional intensity maps only. The problem now lies in converting a multi-dimensional array(**Multi channel image**) to a 3 dimensional array (Often called a **3 channel image**). This problem is solved by stacking algorithms.

There are various stacking algorithms that perform this task, and we go over 2: **Lupton stacking**, and an arbitrary stacking which I have termed **weighted mean stacking** where you, the programmer, determine how much each wavelength contributes to the image

### Stacking with python

The modules we will be using are

- numpy – to manipulate, edit and stack arrays
- astropy – to read fits files and a function for lupton stacking algorithm
- matplotlib – to preview arrays/images
- scipy – to use the Gaussian filter function (to remove noise)
- open-cv – to write data to image files

The data is read from .fits files (Flexible Image Transport System) which is a digital file format designed by NASA to read and store image data and meta data. It is the most common file used in astronomy.

```python
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import cv2 as cv
from astropy.visualization import make_lupton_rgb
```


Now lets store the image files in an array. Be sure to append the `r` infront of every string and double the `\` to `\\` since python considers it as an escape sequence. Add the path of every file into this array, stored within double quotes `"`
```python
image_list = [r"path\\to\\file", r"path\\to\\file"]
```
if you are using Linux/macOS; it is a lot more straightforward
```python
image_list = ["path/to/file", "path/to/file"]
```


Now the files are read using the `fits.getdata(image)` and are stored into another array. We then normalize the values such that the intensity values are from 0-1.
Since the data has a lot of blank-space, we **clip** the data from points 1000 -> 7000 in both X and Y directions. These numbers can be changed depending on your data
```python
images = [fits.getdata(image) for image in image_list]

norm_images = [img[1000:7000, 1000:7000]/ np.percentile(img[1000:7000, 1000:7000], 99) for img in images]
```

### Weighted mean
this n-dimensional data is now converted to 3D array. Since data from lower-wavelength to contribute towards blueish colours and ones from longer wavelengths to contribute towards reddish colours, we map the lower wavelength filters to Blue and higher wavelengths towards Red.
	The numbers used here may vary, choose them according the order in-which you have passed your data.
	The constants that are multiplied with the arrays are weights that can be adjusted accordingly
```python
R = norm_images[5] * 0.7 + norm_images[6] * 0.3
G = norm_images[3] * 0.7 + norm_images[4] * 0.3
B = norm_images[2] * 0.5 + norm_images[1] * 0.25 + norm_images[0] * 0.25

rgb_image = np.stack([R, G, B], axis=-1)
rgb_image = gaussian_filter(rgb_image, sigma=1)
rgb_image = np.clip(rgb_image, 0, 1)
rgb_image_8bit = (rgb_image * 255).astype(np.uint8)
```
We stack the R, G, B images. The `gaussian_filter()` function removes data that deviates too much from the mean. `sigma=1` implies data past the standard deviation is rejected. This value can be increased or decreased to control the noise levels. but is best kept at 1.

The image is previewed with matplotlib
```python
plt.imshow(rgb_image, origin='lower')
plt.axis('off')
```
The clip values (from norm_images) can be adjusted to ensure the entire image is visible.

Once previewed, the image is written to a image file using opencv
```python
cv.imwrite('stacked_image.png', rgb_image_8bit)
```
This image is stored in your directory, and is accessed via your file explorer

### Lupton Stacking
This is a stacking algorithm developed by Lupton et. al (2004). (Read the paper at https://arxiv.org/abs/astro-ph/0312483). It takes in a 3 channel image and generates a colour-corrected image with a natural index. This prevents bright stars from looking overwashed white, while preserving and showing bright objects within the field of view.

Here, we alter the RGB array values
```python
R = norm_images[5] #data from filter within ~620-700 nm
G = norm_images[3] #data from filter around ~550 nm
B = norm_images[0] #data from filter within ~450 nm

rgb_image = make_lupton_rgb(R, G, B, stretch=1, Q=1)
```
`stretch` and `Q` values determine the stretching of colour and its intensity.

Now we write the image to a file using opencv
```python
p0, p100 = np.percentile(rgb_image, (0, 100))
rgb_image = np.clip((rgb_image - p0) / (p100 - p0), 0, 1)
rgb_image = np.flipud(rgb_image)
rgb_image_8bit = (rgb_image*255).astype(np.uint8)
rgb_image_bgr = cv.cvtColor(rgb_image_8bit, cv.COLOR_RGB2BGR)
cv.imwrite('fullstack_lupton.png', rgb_image_bgr)
```
You can view the image from your file explorer


## Photometry
Photometry is the science of measuring the brightness (or flux) of astronomical objects, like stars, galaxies, and nebulae, as they appear in the sky. This measurement is essential for understanding the physical properties of these objects, including their luminosity, distance, temperature, and composition.

Photometry is often conducted on star clusters, and galaxies to determine the distribution of stars (and star-clusters), their ages, and elements within them. This is done by analysing and identifying fits data in a specific wavelength. Photometry from multiple filters can then be combined into a Hertzsprung-Russel Diagram.

### Photometry with python
Modules:
- photutils for identifying stars and calculating their magnitudes.
- acstools to calculate zero_point
- seaborn for data visualization

Open the fits file
```python
from astropy.io import fits

ff = fits.open('path/to/file.fits')
print(ff)
```

View the data
```python
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

plt.figure()
plt.imshow(im_data, origin='lower', norm=LogNorm(), cmap='Greys')
plt.colorbar()
plt.show()
```
To see dimensions of image:
```python
ff.info()
```

Based on the preview of the data, snip the data to include only the relevant data and reject blank data. Be sure to change the values in `im_data[~:~, ~:~]'
```python
section1 = im_data[2000:8000, 2000:8000]
plt.figure()
plt.imshow(section1, origin='lower', norm=LogNorm(), cmap='Greys')
plt.colorbar()
plt.show()
```

Calculate mean,median and standard deviation
```python 
from astropy.stats import sigma_clipped_stats

mean, median, std = sigma_clipped_stats(section1, sigma= 3.0)
print(mean, median, std)
```

Use photutils to identify and mark stars with red rings of radius 5px
```python
from photutils.detection import DAOStarFinder
import numpy as np
from photutils.aperture import CircularAperture


daofind = DAOStarFinder(fwhm=3.0, threshold=5.0*std)
sources1 = daofind(section1 - median)
for col in sources1.colnames:
	if col not in ('id', 'npix'):
		sources1[col].info.format = '%.2f'


positions = np.transpose((sources1['xcentroid'], sources1['ycentroid']))
apertures = CircularAperture(positions, r=5.0)
print(apertures)

plt.imshow(section1, cmap='Greys', origin='lower', norm=LogNorm(), interpolation='nearest')
apertures.plot(color='red', lw=1.5, alpha=0.5);

```

Create an annulus of 5 px width to calculate background radiation and display it
``` python
from photutils.aperture import CircularAnnulus, ApertureStats, aperture_photometry

annulus_aperture = CircularAnnulus(positions, r_in=10, r_out=15)
plt.imshow(section1, cmap='Greys', norm=LogNorm(), origin='lower', interpolation='nearest')
apertures.plot(color='red', lw=1.5, alpha=0.5);
annulus_aperture.plot(color='yellow', lw=1.5, alpha=0.3);
```

Calculate the radiation within the apertures and the annulus. Radiation within the aperture is considered to be from the star, while radiation from the annulus is background radiation
```python
aper_stats = ApertureStats(section1, annulus_aperture)
bkg_mean = aper_stats.mean
aper_area = apertures.area_overlap(section1)
total_bkg = bkg_mean*aper_area
star_data = aperture_photometry(section1, apertures)

star_data['total_bkg'] = total_bkg
for col in star_data.colnames:
    star_data[col].info.format = '%.8g'

star_data.pprint()
```

Now, we need to find 3 important data that is bundled with the file. This can be seen by reading the metadata of the .fits file. 
Open terminal in the directory where the file is stored, and read the meta data with this bash command
```bash
user@device$ more file-name.fits
```
Note down **date, exposure time, instrument**(usually WFC) and **filter**
Now run this in python
```python
from acstools import acszpt

date = '2006-05-31'
instrument = 'WFC'
filter = 'F435W'
exposure_time = 1800

q = acszpt.Query(date= date, detector = instrument)
zpt_table = q.fetch()
q_filter = acszpt.Query(date=date, detector=instrument,filt=filter)
filter_zpt=q_filter.fetch()

print(filter_zpt)
type(filter_zpt)
```
Remember to replace the variables with your values
The value under `ABmag` is the zero point value, be sure to note it down

The following code block now calculates the apparent magnitude of stars. This data is used for making HR diagrams and spectral plots
```python
import math
zeropoint = 25.674
exposure_time = 1800

magnitudes = []
for line in star_data:
    magnitudes.append(zeropoint-(2.4*math.log10(abs(line[3]-line[4])/exposure_time)))
star_data['magnitude'] = magnitudes

star_data.pprint(max_lines=-1, max_width=-1)
```

View the spectral distribution with
```python
import seaborn as sns
import pandas as pd

mag_min = np.min(magnitudes)
mag_max = np.max(magnitudes)
mag_mid = np.mean(magnitudes)
mag_std = np.std(magnitudes)

print(f'{mag_min}\t{mag_max}\t{mag_mid}\t{mag_std}')

plt.style.use('dark_background')
mag_df = pd.DataFrame(magnitudes, columns=['Magnitude'])
fig, ax1 = plt.subplots()
ax1.set_ylabel('Density')

sns.histplot(data=mag_df, x='Magnitude', binwidth=0.5, color='orange', ax=ax1)
ax2 = ax1.twinx()

sns.kdeplot(data=mag_df, x='Magnitude', bw_adjust=0.25, color='blue', ax=ax2)
ax2.set_ylabel('frequency')
ax2.grid(False)
plt.show()
```


## Hertzsprung-Russel diagrams
H-R diagram is a scatter plot that plots stars according to two primary characteristics: **luminosity (or magnitude)** and **surface temperature (or spectral type).**

Import all relevant modules
```python
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
import numpy as np 
from photutils.aperture import CircularAperture
from photutils.aperture import CircularAnnulus, ApertureStats, aperture_photometry
from acstools import acszpt
import math
import seaborn as sns
import pandas as pd
```

Calculating apparent magnitude for 814nm (refer `hrdiagram.ipynb`)
```python

ff = fits.open('ngc6101/hlsp_hugs_hst_acs-wfc_ngc6101_f814w_v1_stack-0340s.fits')
# ~~~ for full code refer to hrdiagram.ipynb
data_f814w['magnitude'] = magnitudes
data_f814w.pprint(max_lines=-1, max_width=-1)
```

Calculating apparent magnitude for 606nm (refer `hrdiagram.ipynb`)
```python

ff = fits.open('ngc6101/hlsp_hugs_hst_acs-wfc_ngc6101_f606w_v1_stack-0370s.fits')
ff.info()
# ~~~ for full code refer to hrdiagram.ipynb
data_f606w['magnitude'] = magnitudes
data_f606w.pprint(max_lines=-1, max_width=-1)
```

We come upon an issue here; the 2 data sets dont have an equal number of rows. This is to be solved by retaining stars that are present in both datasets. This is possible by comparing `xcenter` and `ycenter` and keeping those present in both with a small margin of variation.
- The issue is the time-complexity of this operation. Comparing both of the data sets is less process intense but takes O(n\*m) time
- Instead, it is faster to map the coordinates onto a k-D tree for both datasets, determine tolerance and then match them up to produce a new dataset. This is considerably faster and has a time-complexity of O(n logn + m logm) time.
- The 2 matched datasets are then subtracted to get data for spectral range of 814nm - 606nm
```python
from astropy.table import QTable, join

'''
matched_data = join(data_f606w, data_f814w, keys=['xcenter', 'ycenter'], join_type='inner')
print(matched_data)
'''

from scipy.spatial import cKDTree
from astropy.table import QTable

# Build KDTree for fast spatial matching
coords_f606w = np.vstack([data_f606w['xcenter'], data_f606w['ycenter']]).T
coords_f814w = np.vstack([data_f814w['xcenter'], data_f814w['ycenter']]).T

# Build KDTree for the second dataset
tree_f814w = cKDTree(coords_f814w)

# Define the tolerance for matching coordinates
tolerance = 0.5  # Adjust based on precision

# Query the tree to find matches within the tolerance
distances, indices = tree_f814w.query(coords_f606w, distance_upper_bound=tolerance)

# Filter valid matches (where distances are within the tolerance)
valid_matches = distances < tolerance

# Get the matching rows
matched_f606w = data_f606w[valid_matches]
matched_f814w = data_f814w[indices[valid_matches]]

magnitude_difference = matched_f814w['magnitude'] - matched_f606w['magnitude']
    
# Add the magnitude difference as a new column to one of the matched tables (e.g., matched_f814w)
matched_f814w['magnitude_difference'] = magnitude_difference

print(matched_f814w[['magnitude', 'magnitude_difference']])
```

This processed data is used to plot the H-R diagram
```python
import matplotlib.pyplot as plt

# Prepare data for the HR diagram
magnitudes_f606w = matched_f606w['magnitude']  # Magnitude from matched_f606w
magnitude_difference = matched_f814w['magnitude_difference']  # Difference from matched_f814w

# Create a figure and axis with dark background
plt.style.use('dark_background')  # Set the style to dark background
plt.figure(figsize=(10, 10))

# Scatter plot with a different marker (e.g., 'D' for diamond)
plt.scatter(magnitude_difference, magnitudes_f606w, color='red', marker='*', alpha=0.7)

# Set labels and title
plt.xlabel('Magnitude Difference (F814W - F606W)', color='white')
plt.ylabel('Magnitude (F606W)', color='white')
plt.title('Hertzsprung-Russell Diagram', color='white')

# Invert the y-axis for magnitudes (so lower values are at the top)
plt.gca().invert_yaxis()

# Set axis limits if necessary
#plt.xlim(min(magnitude_difference) - 1, max(magnitude_difference) + 1)
#plt.ylim(min(magnitudes_f606w) - 1, max(magnitudes_f606w) + 1)

# Add grid for better readability
plt.grid(color='gray')

# Show the plot
plt.show()
```
