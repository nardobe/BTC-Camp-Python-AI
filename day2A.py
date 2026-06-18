import matplotlib.pyplot as plt #2
import urllib.request, zipfile, os, shutil #3
import numpy as np #4
import random #6
# Images are Numbers Lesson:
# 1-----------------grid------------------
grid = [
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0],
]

print(grid[0])      # the first ROW
print(grid[0][1])   # row 0, column 1
print(grid[2][2])   # row 2, column 2
# 2--------------blocks in grid---------------------
smiley = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
]

plt.imshow(smiley, cmap='gray')
plt.title('My first image: an 8x8 grid of numbers!')
plt.show()

my_art = [
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
]

import matplotlib.pyplot as plt
plt.imshow(my_art, cmap='gray')
plt.show()
# 3----------------images--------------------
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open('cats_and_dogs_filtered/train/cats/cat.190.jpg')
plt.imshow(img)
plt.title('A real photo')
plt.show()
# 4---We can see what the computer sees by converting the photo into a numpy array---
pixels = np.array(img)
print("Shape of this image:", pixels.shape) #shape-> (height, width, 3 (rgb))
print("Total number in this one photo:", pixels.size)
# inspect one pixel:
y, x = 350, 200
r, g, b = pixels[y, x]
print(f"Pixel at row {y}, column {x}:  Red={r}  Green={g}  Blue={b}")

# Show that exact color as a little square
plt.figure(figsize=(1.5, 1.5))
plt.imshow([[pixels[y, x]]])
plt.axis("off")
plt.title(f"RGB = ({r}, {g}, {b})")
plt.show()
# 5-------Photo math/manipulation-----------------------
fig, ax = plt.subplots(1, 4, figsize=(16, 4))

ax[0].imshow(pixels);                 ax[0].set_title('Original')
ax[1].imshow(49 - pixels);           ax[1].set_title('49 - pixels  (inverted!)')
ax[2].imshow(pixels[50:, ::-1]);        ax[2].set_title('Columns reversed (mirror)')
ax[3].imshow(pixels[::8, ::8]);       ax[3].set_title('Every 8th pixel (low-res)')

for a in ax: a.axis('off')
plt.show()
plt.imshow(pixels[::-1])
plt.title(f"rows reversed")
plt.show()
# 6 ----------cats & dogs - the 40 year predicament-----------------
cat_dir = 'cats_and_dogs_filtered/train/cats'
dog_dir = 'cats_and_dogs_filtered/train/dogs'

fig, ax = plt.subplots(2, 4, figsize=(16, 8))
for i in range(4):
    cat = np.array(Image.open(os.path.join(cat_dir, random.choice(os.listdir(cat_dir)))))
    dog = np.array(Image.open(os.path.join(dog_dir, random.choice(os.listdir(dog_dir)))))
    ax[0, i].imshow(cat); ax[0, i].set_title(f'cat — {cat.shape[0]}x{cat.shape[1]} pixels')
    ax[1, i].imshow(dog); ax[1, i].set_title(f'dog — {dog.shape[0]}x{dog.shape[1]} pixels')
for a in ax.flat: a.axis('off')
plt.show()

print('Average pixel value of last cat photo:', np.array(cat).mean().round(1))
print('Average pixel value of last dog photo:', np.array(dog).mean().round(1))