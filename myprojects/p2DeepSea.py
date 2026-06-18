import importlib, subprocess, sys
import tensorflow as tf #1
import numpy as np #1
import matplotlib.pyplot as plt #1
from tensorflow.keras import layers, models #4
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay #8
import glob #8
from PIL import Image #8
packages = {"numpy": "numpy", "matplotlib": "matplotlib", "PIL": "pillow", "sklearn": "scikit-learn", "tensorflow": "tensorflow"}
for mod, pip_name in packages.items():
    try:
        importlib.import_module(mod)
        print(pip_name, 'is ready')
    except ImportError:
        print('Installing', pip_name, '(one time, please wait)...')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pip_name])
        print(pip_name, 'installed')
print('\nAll set')
TEAM = 'Team Kyogre'
PROJECT = 'Deep Sea Detective'
print(f'{TEAM} presents: {PROJECT}')
# 1--------load pictures in-------------
(xtr, ytr), (xte, yte) = tf.keras.datasets.cifar100.load_data(label_mode='fine')

MY_TEAMS = {'fish': [73, 67, 91, 32, 1], 'mammal': [30, 95, 72, 55, 4]}
# ^ 2 teams of animals. The AI learns to tell the TEAMS apart

class_names = list(MY_TEAMS)
id2team = {fid: gi for gi, g in enumerate(class_names) for fid in MY_TEAMS[g]}
NUM_CLASSES = len(class_names)
GRAY = False
experiments = {}                                   # the tracker starts empty

def grab(x_all, y_all, n):                         # collect up to n photos of each chosen species
    xo, yo, c = [], [], {fid: 0 for fid in id2team}
    for img, lab in zip(x_all, y_all.flatten()):
        if lab in id2team and c[lab] < n:
            xo.append(img); yo.append(id2team[lab]); c[lab] += 1
    return np.array(xo), np.array(yo)

x_train, y_train = grab(xtr, ytr, 2000)
x_test,  y_test  = grab(xte, yte, 600)

# MobileNet likes bigger pictures, so enlarge ours to 96x96 (one line).
x_train = tf.image.resize(tf.cast(x_train, 'float32'), (96, 96)).numpy().astype('uint8')
x_test  = tf.image.resize(tf.cast(x_test,  'float32'), (96, 96)).numpy().astype('uint8')
print('The AI will learn:', class_names[0], 'VS', class_names[1])
print('Study photos:', len(x_train), '| Exam photos:', len(x_test))
# 2-------------check-----------------------
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
# 3----------------set brain dials---------------
EPOCHS        = 20
HIDDEN_UNITS  = 128
DROPOUT       = 0.30
LEARNING_RATE = 0.001
print('Dials set →', dict(EPOCHS=EPOCHS, HIDDEN_UNITS=HIDDEN_UNITS, DROPOUT=DROPOUT, LEARNING_RATE=LEARNING_RATE))
# 4----------------------Train AI and record------------
RUN_NAME = 'run 1: starting point'   # Rename each time theres a change on a dial!

# (First run downloads MobileNet once — needs internet — then it's cached.)
base = tf.keras.applications.MobileNetV2(input_shape=(96, 96, 3), include_top=False,
                                         pooling='avg', weights='imagenet')
base.trainable = False                                  # keep the expert eyes frozen

model = models.Sequential([
    layers.Input(shape=(96, 96, 3)),
    layers.Rescaling(1 / 127.5, offset=-1),             # scale pixels the way MobileNet expects
    base,                                               # the frozen expert eyes
    layers.Dense(HIDDEN_UNITS, activation='relu'),      # the decision layer
    layers.Dropout(DROPOUT),
    layers.Dense(NUM_CLASSES, activation='softmax'),    # one score per team, adds to 100%
])
model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                    epochs=EPOCHS, batch_size=32, verbose=2)
score = history.history['val_accuracy'][-1] * 100
experiments[RUN_NAME] = round(score, 1)
print(f'\n {RUN_NAME}  ->  {score:.1f}%'); print('Experiments so far:', experiments)
# 5----------------confidence score-------------
preds = model.predict(x_test, verbose=0)[:, 1]      # P(mammal)
sample = np.random.choice(len(x_test), 8, replace=False)
plt.figure(figsize=(15, 4))
for k, idx in enumerate(sample):
    pct = preds[idx] * 100
    flag = '🐬' if pct >= 50 else '🐟'
    plt.subplot(1, 8, k + 1); plt.imshow(x_test[idx]); plt.axis('off')
    plt.title(f'{flag} {pct:.0f}%', fontsize=11)
plt.suptitle('mammal score for 8 random animals (higher = more mammal)', fontsize=12)
plt.tight_layout(); plt.show()
# 6---------------------confusion matrix----------------
pred_labels = model.predict(x_test, verbose=0).argmax(axis=1)
cm = confusion_matrix(y_test, pred_labels)
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap='Greens', colorbar=False)
plt.title('The AI\'s confusion matrix'); plt.tight_layout(); plt.show()
# 7-----------------experiment tracker------------------
plt.figure(figsize=(9, max(2, 0.6 * len(experiments))))
names = list(experiments); vals = [experiments[n] for n in names]
plt.barh(names, vals, color='#2C5F2D')
for i, v in enumerate(vals): plt.text(v + 0.5, i, f'{v}%', va='center', fontsize=11)
plt.xlim(0, 100); plt.xlabel('Exam accuracy (%)'); plt.title('Our tuning experiments')
plt.gca().invert_yaxis(); plt.tight_layout(); plt.show()
best = max(experiments, key=experiments.get)
print(f'🥇 Best so far: "{best}" at {experiments[best]}%')
# 8---------------real photo tester------------------------
files = sorted(glob.glob('my_tests/*'))
if not files:
    print('No photos yet')
else:
    imgs = np.array([np.array(Image.open(f).convert('RGB').resize((96, 96))) for f in files])
    scores = model.predict(imgs, verbose=0)[:, 1]
    n = len(files)
    plt.figure(figsize=(min(15, 2.6 * n), 3.2))
    for k, f in enumerate(files):
        verdict = class_names[1] if scores[k] >= 0.5 else class_names[0]
        plt.subplot(1, n, k + 1); plt.imshow(Image.open(f).convert('RGB')); plt.axis('off')
        plt.title(f'{verdict}\n{scores[k]*100:.0f}%', fontsize=11)
    plt.suptitle('The AI judging real photos from the internet', fontsize=13)
    plt.tight_layout(); plt.show()
# specific img----------
from pathlib import Path

# Change these
image_folder = "my_tests"
image_name = "blob.jpg"
# ---------------

folder = Path(image_folder)
files = [folder / image_name]

files = [f for f in files if f.exists()]

if not files:
    print("Image not found.")
    print("Folder:", folder)
    print("Image name:", image_name)

else:
    imgs = np.array([
        np.array(Image.open(f).convert("RGB").resize((96, 96)))
        for f in files
    ])

    # If your training images were divided by 255, uncomment line:
    # imgs = imgs / 255.0

    predictions = model.predict(imgs, verbose=0)

    n = len(files)
    plt.figure(figsize=(min(15, 2.8 * n), 3.4))

    for k, f in enumerate(files):
        probs = predictions[k]

        predicted_index = np.argmax(probs)
        verdict = class_names[predicted_index]
        confidence = probs[predicted_index]

        plt.subplot(1, n, k + 1)
        plt.imshow(Image.open(f).convert("RGB"))
        plt.axis("off")
        plt.title(f"{f.name}\n{verdict}\n{confidence * 100:.0f}%", fontsize=10)

    plt.suptitle("judgement", fontsize=13)
    plt.tight_layout()
    plt.show()

    print("Raw model prediction:")
    print(predictions)

    print("\nClass names:")
    print(class_names)