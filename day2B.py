import urllib.request, zipfile, os, shutil # 1
import tensorflow as tf # 1
import matplotlib.pyplot as plt # 2
import numpy as np # 5
from pathlib import Path # 6
from PIL import Image # 6
# Build first AI - cat vs dog.
# 1-----------Get data/examples for ML------------
# check first:
print("TensorFlow version:", tf.__version__)
print("GPU detected:" , "YES ⚡ (fast!)" if tf.config.list_physical_devices("GPU") else "no — not a problem, training just takes a few minutes")

train_dir = "cats_and_dogs_filtered/train"
val_dir   = "cats_and_dogs_filtered/validation"

for folder in ["cats", "dogs"]:
    print(f"Training {folder}: {len(os.listdir(os.path.join(train_dir, folder)))} photos")
for folder in ["cats", "dogs"]:
    print(f"Validation {folder}: {len(os.listdir(os.path.join(val_dir, folder)))} photos")
# 2-----------------------(Keras is our AI library)-------------
IMG_SIZE = (96, 96)   # every photo gets resized to 96x96 pixels — small and fast
BATCH_SIZE = 64       # the AI studies photos in groups of 64

train_ds = tf.keras.utils.image_dataset_from_directory(   # load the study material:
    train_dir,                # read photos from the train/ folder
    image_size=IMG_SIZE,      # resize every photo to 96x96
    batch_size=BATCH_SIZE,    # hand them to the AI 64 at a time
    shuffle=True)             # mix cats and dogs so it never sees all cats first

val_ds = tf.keras.utils.image_dataset_from_directory(     # load the exam (never studied!):
    val_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE)

class_names = train_ds.class_names        # the folder names become the labels
print("Classes the AI will learn:", class_names)

plt.figure(figsize=(12, 6))
for images, labels in train_ds.take(1):
    for i in range(8):
        ax = plt.subplot(2, 4, i + 1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[labels[i]])
        plt.axis('off')
plt.show()
# 3------------Build brain layer by layer through CNN---------------
# copied:
model = tf.keras.Sequential([                          # a stack of layers, read top to bottom
    tf.keras.layers.Input(shape=(96, 96, 3)),             # photos come in: 96x96 pixels, 3 colors (R,G,B)
    tf.keras.layers.Rescaling(1./255),                    # pixel values 0-255 become 0-1

    tf.keras.layers.Conv2D(32, 3, activation="relu"),     # 32 pattern detectors, each looks at 3x3 pixels
    tf.keras.layers.MaxPooling2D(),                       # shrink the image in half, keep strongest signals

    tf.keras.layers.Conv2D(64, 3, activation="relu"),     # 64 detectors for BIGGER patterns (fur, ears)
    tf.keras.layers.MaxPooling2D(),                       # shrink again

    tf.keras.layers.Conv2D(128, 3, activation="relu"),    # 128 detectors for even bigger patterns (faces)
    tf.keras.layers.MaxPooling2D(),                       # shrink again

    tf.keras.layers.Flatten(),                            # unroll the grid into one long list of numbers
    tf.keras.layers.Dense(128, activation="relu"),        # 128 neurons combine all the clues
    tf.keras.layers.Dropout(0.3),                         # while studying, randomly ignore 30% of neurons
    tf.keras.layers.Dense(1, activation="sigmoid"),       # final answer: ONE number. 0 = cat ... 1 = dog
])

model.compile(                                            # get the model ready to learn:
    optimizer="adam",                                     # the algorithm that adjusts the knobs after each guess
    loss="binary_crossentropy",                           # how "wrongness" is measured for a 2-choice question
    metrics=["accuracy"],                                 # also report the % of photos guessed correctly
)

model.summary()                                           # print the stack — see how the image shrinks layer by layer
# 4-----------Train through epochs to inc accuracy----------------------------
EPOCHS = 8    # one epoch = the AI studies every training photo once

history = model.fit(       # learning moment:
    train_ds,              # study data (4,000 labeled photos)
    validation_data=val_ds,# the "exam", re-taken after every epoch
    epochs=EPOCHS,         # how many times to study the whole pile
)
# copied: shows accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.figure(figsize=(8, 5))
plt.plot(acc, 'o-', label='Training accuracy (study material)')
plt.plot(val_acc, 's-', label='Validation accuracy (the exam)')
plt.xlabel('Epoch'); plt.ylabel('Accuracy')
plt.title('Your AI learning to tell cats from dogs')
plt.legend(); plt.grid(alpha=0.3); plt.ylim(0.5, 1.0)
plt.show()

print(f'Final exam score: {val_acc[-1]*100:.1f}% on 1,000 photos the AI never saw before!')
# 5--------predictions with confidence--------------
for images, labels in val_ds.take(1):
    probs = model.predict(images, verbose=0).flatten()   # 0.0 = definitely cat, 1.0 = definitely dog

    plt.figure(figsize=(14, 7))
    for i in range(8):
        ax = plt.subplot(2, 4, i + 1)
        plt.imshow(images[i].numpy().astype('uint8'))
        pred = class_names[int(probs[i] > 0.5)]
        conf = probs[i] if probs[i] > 0.5 else 1 - probs[i]
        truth = class_names[labels[i]]
        color = 'green' if pred == truth else 'red'
        plt.title(f'AI says: {pred} ({conf*100:.0f}% sure)', color=color)
        plt.axis('off')
    plt.show()
# 6------fun test-----------------
# check:
print("Notebook is running from:")
print(Path.cwd())

folder_path = Path(r"my_photos") # folder where imgs saved
image_name = "img1.jpg" # exact image file name, including .jpg/.png
image_name2 = "img2.jpg"
image_name3 = "img3.jpg"


fname = folder_path / image_name # joins folder path + image name
fname2 = folder_path / image_name2
fname3 = folder_path / image_name3

print(f"Opening: {fname}")
print(f"Opening: {fname2}")
print(f"Opening: {fname3}")

img = Image.open(fname).convert("RGB").resize(IMG_SIZE) # open and resize image
img2 = Image.open(fname2).convert("RGB").resize(IMG_SIZE)
img3 = Image.open(fname3).convert("RGB").resize(IMG_SIZE)

arr = np.array(img)[np.newaxis, ...].astype("float32") # convert image to model input
arr = np.array(img2)[np.newaxis, ...].astype("float32")
arr = np.array(img3)[np.newaxis, ...].astype("float32")


prob_dog = model.predict(arr, verbose=0)[0][0] # get dog probability

pred = class_names[int(prob_dog > 0.5)] # choose class based on probability
conf = prob_dog if prob_dog > 0.5 else 1 - prob_dog

plt.figure()
plt.imshow(img)
plt.axis("off")
plt.title(f"AI says: {pred} — {conf*100:.0f}% confident")
plt.show()


plt.figure()
plt.imshow(img2)
plt.axis("off")
plt.title(f"AI says: {pred} — {conf*100:.0f}% confident")
plt.show()

plt.figure()
plt.imshow(img3)
plt.axis("off")
plt.title(f"AI says: {pred} — {conf*100:.0f}% confident")
plt.show()
