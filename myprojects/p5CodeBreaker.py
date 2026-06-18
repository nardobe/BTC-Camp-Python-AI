import importlib, subprocess, sys
packages = {"numpy": "numpy", "matplotlib": "matplotlib", "PIL": "pillow", "sklearn": "scikit-learn", "tensorflow": "tensorflow", "cv2": "opencv-python"}
for mod, pip_name in packages.items():
    try:
        importlib.import_module(mod)
        print(pip_name, 'is ready')
    except ImportError:
        print('Installing', pip_name, '(one time, please wait)...')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pip_name])
        print(pip_name, 'installed')
TEAM = 'Team Hakkers'
PROJECT = 'Secret C0de Breaker'
# 1------load data------------------
print(f'{TEAM} presents: {PROJECT}')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(xtr, ytr), (xte, yte) = tf.keras.datasets.mnist.load_data()

class_names = [str(d) for d in range(10)]
NUM_CLASSES = len(class_names)
INPUT_SHAPE = (28, 28, 1)
GRAY = True
experiments = {}

def subsample(x_all, y_all, n):
    xo, yo, c = [], [], {i: 0 for i in range(NUM_CLASSES)}
    for img, lab in zip(x_all, y_all.flatten()):
        if c[lab] < n:
            xo.append(img); yo.append(lab); c[lab] += 1
    return np.array(xo)[..., np.newaxis], np.array(yo)

x_train, y_train = subsample(xtr, ytr, 1500)
x_test,  y_test  = subsample(xte, yte, 400)
print('Classes:', class_names, '| Study:', len(x_train), '| Exam:', len(x_test))
# 2---------check-----------
cmap = 'gray' if GRAY else None
plt.figure(figsize=(2.3 * NUM_CLASSES, 4.6))
for col, name in enumerate(class_names):
    members = np.where(y_train == col)[0]
    for row in range(2):
        plt.subplot(2, NUM_CLASSES, row * NUM_CLASSES + col + 1)
        plt.imshow(x_train[members[row]].squeeze(), cmap=cmap)
        if row == 0: plt.title(name, fontsize=10)
        plt.axis('off')
plt.suptitle('A few examples of each group', y=1.02, fontsize=12)
plt.tight_layout(); plt.show()
# 3---------setting dials------------
EPOCHS        = 10
BATCH_SIZE    = 64
LEARNING_RATE = 0.001
DROPOUT       = 0.30
AUGMENT       = False
EXTRA_CONV    = True
print('Dials set →', dict(EPOCHS=EPOCHS, BATCH_SIZE=BATCH_SIZE, LEARNING_RATE=LEARNING_RATE, DROPOUT=DROPOUT, AUGMENT=AUGMENT, EXTRA_CONV=EXTRA_CONV))
# 4----------train AI------------------
from tensorflow.keras import layers, models
RUN_NAME = 'run 1: starting point' 

def build_model():
    net = [layers.Input(shape=INPUT_SHAPE), layers.Rescaling(1./255)]
    if AUGMENT:
        net += [layers.RandomFlip('horizontal'), layers.RandomRotation(0.10), layers.RandomZoom(0.10)]
    net += [layers.Conv2D(32, 3, padding='same', activation='relu'), layers.BatchNormalization(), layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'), layers.BatchNormalization(), layers.MaxPooling2D()]
    if EXTRA_CONV:
        net += [layers.Conv2D(128, 3, padding='same', activation='relu'), layers.BatchNormalization(), layers.MaxPooling2D()]
    net += [layers.Flatten(), layers.Dense(128, activation='relu'), layers.Dropout(DROPOUT),
            layers.Dense(NUM_CLASSES, activation='softmax')]
    m = models.Sequential(net)
    m.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m

model = build_model()
history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                    epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)
score = history.history['val_accuracy'][-1] * 100
experiments[RUN_NAME] = round(score, 1)
print(f'\n {RUN_NAME}  ->  {score:.1f}%'); print('Experiments so far:', experiments)
# 5----------AI@work-----------------
cmap = 'gray' if GRAY else None
preds = model.predict(x_test, verbose=0)
plt.figure(figsize=(14, 6))
for i in range(12):
    plt.subplot(2, 6, i + 1); plt.imshow(x_test[i].squeeze(), cmap=cmap)
    guess = class_names[preds[i].argmax()]; truth = class_names[y_test[i]]
    plt.title(f'{guess}\n{preds[i].max()*100:.0f}%', fontsize=9, color='green' if guess == truth else 'red')
    plt.axis('off')
plt.tight_layout(); plt.show()
# 6---------------secret code------------
SECRET = [4, 2, 0, 5, 1]       # make your own secret code (any single digits 0-9)
read = []
plt.figure(figsize=(2 * len(SECRET), 2.4))
for k, digit in enumerate(SECRET):
    idx = np.where(y_test == digit)[0][0]
    guess = int(model.predict(x_test[idx][np.newaxis], verbose=0)[0].argmax())
    read.append(guess)
    plt.subplot(1, len(SECRET), k + 1); plt.imshow(x_test[idx].squeeze(), cmap='gray'); plt.axis('off')
    plt.title(str(guess), fontsize=14)
plt.suptitle('What the AI read from the handwriting'); plt.show()
print('Secret code:', ''.join(map(str, SECRET)))
print('AI read:    ', ''.join(map(str, read)))
print('CODE CRACKED!' if read == SECRET else 'the AI misread a digit')
# 7-------------confusion matrix--------------
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
pred_labels = model.predict(x_test, verbose=0).argmax(axis=1)
cm = confusion_matrix(y_test, pred_labels)
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap='Greens', colorbar=False, xticks_rotation=45)
plt.title('Where does the AI get confused?'); plt.tight_layout(); plt.show()
# 8-------experiment tracker-----------------
plt.figure(figsize=(9, max(2, 0.6 * len(experiments))))
names = list(experiments); vals = [experiments[n] for n in names]
plt.barh(names, vals, color='#2C5F2D')
for i, v in enumerate(vals): plt.text(v + 0.5, i, f'{v}%', va='center', fontsize=11)
plt.xlim(0, 100); plt.xlabel('Exam accuracy (%)'); plt.title('Our tuning experiments')
plt.gca().invert_yaxis(); plt.tight_layout(); plt.show()
best = max(experiments, key=experiments.get)
print(f'🥇 Best so far: "{best}" at {experiments[best]}%')
# 9----------webcam photo-----------------
# Webcam helper — grabs ONE photo, but waits so you can get ready.
# write one large dark digit on WHITE paper and hold it up.

def snap_photo(delay=5, camera_index=0):
    try:
        import cv2
        import time
    except ImportError:
        print("OpenCV missing — run this once, then restart the kernel:")
        print("import sys")
        print("!{sys.executable} -m pip install opencv-python")
        return None

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("No webcam found — try camera_index=1 or use the file option instead.")
        cap.release()
        return None

    # Ask for better resolution. Some webcams may ignore this, and that is okay.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Warm up the camera so exposure/focus improves before the final photo.
    ok, frame = False, None
    for _ in range(30):
        ok, frame = cap.read()

    print("Get ready!!")
    print("Write ONE big dark digit on plain white paper.")
    print("Hold it flat and centered in front of the camera.")
    print("Taking photo in:")

    for s in range(delay, 0, -1):
        print(f"{s}...", end=" ", flush=True)
        time.sleep(1)

    print("SNAP!")

    # Grab a few final frames after countdown.
    for _ in range(10):
        ok, frame = cap.read()

    cap.release()

    if not ok or frame is None:
        print("Could not grab a photo — try again.")
        return None

    # OpenCV gives BGR; matplotlib/PIL expect RGB.
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


print("snap_photo() is ready.")
# --
# Snap, crop, convert to clean MNIST style, and read the digit

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os


def center_crop_with_box(image_rgb, crop_fraction=0.70, y_shift=0.00):
    """
    Crop the center square of the webcam image.

    crop_fraction:
        Bigger number = bigger crop.
        Try 0.55, 0.65, 0.70, or 0.80.

    y_shift:
        0.00 = center
        positive = move crop down
        negative = move crop up
    """
    h, w, _ = image_rgb.shape
    size = int(min(h, w) * crop_fraction)

    cx = w // 2
    cy = int(h // 2 + y_shift * h)

    x1 = max(0, cx - size // 2)
    x2 = min(w, cx + size // 2)
    y1 = max(0, cy - size // 2)
    y2 = min(h, cy + size // 2)

    cropped = image_rgb[y1:y2, x1:x2]
    return cropped, (x1, y1, x2, y2)


def make_mnist_style_digit(crop_rgb, min_area=40):
    """
    Convert a real webcam crop into clean MNIST style.

    MNIST expects:
    - black background
    - bright/white digit
    - centered digit
    - 28x28 image

    This version uses OpenCV thresholding to remove gray paper/background noise.
    """
    import cv2

    # RGB -> grayscale
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)

    # Reduce camera/paper noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Make dark handwriting become bright strokes on black background.
    # Otsu chooses the threshold automatically.
    _, bw = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Remove tiny specks/noise, but keep the digit strokes
    kernel = np.ones((2, 2), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)

    # Find digit pixels
    ys, xs = np.where(bw > 0)

    if len(xs) < min_area or len(ys) < min_area:
        print("Could not find a clear digit.")
        print("Try thicker marker, bigger digit, plain white paper, and better lighting.")
        return np.zeros((28, 28), dtype="float32")

    # Tight crop around strokes
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    digit = bw[y1:y2 + 1, x1:x2 + 1]

    # Add padding around digit before making it square
    pad = 10
    digit = np.pad(digit, pad_width=pad, mode="constant", constant_values=0)

    # Pad to square so the digit is not stretched
    h, w = digit.shape
    size = max(h, w)
    square = np.zeros((size, size), dtype=np.uint8)

    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    square[y_offset:y_offset + h, x_offset:x_offset + w] = digit

    # Resize to 20x20, then center in 28x28 like MNIST
    try:
        resample_mode = Image.Resampling.LANCZOS
    except AttributeError:
        resample_mode = Image.LANCZOS

    digit_20 = Image.fromarray(square).resize((20, 20), resample_mode)

    canvas = np.zeros((28, 28), dtype=np.uint8)
    canvas[4:24, 4:24] = np.array(digit_20)

    return canvas.astype("float32")


def read_digit_from_camera_image(image_rgb, crop_fraction=0.70, y_shift=0.00):
    """
    Show original photo, crop the digit area, convert to MNIST style,
    and ask the trained model to read it.
    """
    cropped, box = center_crop_with_box(
        image_rgb,
        crop_fraction=crop_fraction,
        y_shift=y_shift
    )

    x1, y1, x2, y2 = box

    # Show original photo with red crop box
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(image_rgb)
    ax.add_patch(
        patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=3,
            edgecolor="red",
            facecolor="none"
        )
    )
    ax.set_title("Original photo — red box is the digit crop")
    ax.axis("off")
    plt.show()

    # Show cropped region
    plt.figure(figsize=(4, 4))
    plt.imshow(cropped)
    plt.axis("off")
    plt.title("Cropped digit area")
    plt.show()

    # Convert crop into MNIST-style 28x28 image
    arr = make_mnist_style_digit(cropped)

    # Show what the AI actually sees
    plt.figure(figsize=(3, 3))
    plt.imshow(arr, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")
    plt.title("What the AI sees: 28x28 MNIST style")
    plt.show()

    # IMPORTANT:
    # In THIS notebook, the model already has layers.Rescaling(1./255).
    # So DO NOT divide by 255 here.
    # Pass the 0–255 image directly into the model.
    x = arr[np.newaxis, ..., np.newaxis]

    p = model.predict(x, verbose=0)[0]

    prediction = int(p.argmax())
    confidence = float(p.max() * 100)

    plt.figure(figsize=(3, 3))
    plt.imshow(arr, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")
    plt.title(f"AI reads: {prediction} ({confidence:.0f}% sure)", fontsize=14)
    plt.show()

    print("Prediction scores:")
    top3 = np.argsort(p)[-3:][::-1]
    for digit in top3:
        print(f"{digit}: {p[digit] * 100:.1f}%")

    print()
    print("Tips if it is wrong:")
    print("- Use a thick dark marker.")
    print("- Write one large digit, not a thin outline.")
    print("- Use plain white paper.")
    print("- Hold the digit inside the red box.")
    print("- Avoid shadows and tilted paper.")


# Take webcam photo
photo = snap_photo(delay=5)

# Backup file option: save an image as my_digit.png in the same folder as the notebook
if photo is None and os.path.exists("my_digit.png"):
    print("Using my_digit.png instead.")
    photo = np.array(Image.open("my_digit.png").convert("RGB"))

if photo is not None:
    read_digit_from_camera_image(
        photo,
        crop_fraction=0.70,
        y_shift=0.00
    )
else:
    print("Take a webcam photo, or save my_digit.png next to this file, then re-run.")