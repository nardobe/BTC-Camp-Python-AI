import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# load all 6
# CIFAR-10 is built into Keras. First run downloads it once (~picture library), then it's cached.
(x_train_all, y_train_all), (x_test_all, y_test_all) = tf.keras.datasets.cifar10.load_data()

# CIFAR-10 has 10 things. We keep only the 6 ANIMALS (the others are planes, cars, ships, trucks).
ALL_NAMES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
MY_ANIMALS = [3, 4, 5, 7]          # ✏️ bird, cat, deer, dog, frog, horse (these are CIFAR-10 ids) I ONLY CHOSE FURRY ANIMALS

remap = {old: new for new, old in enumerate(MY_ANIMALS)}   # renumber them 0..5
class_names = [ALL_NAMES[i] for i in MY_ANIMALS]           # the names, in order
NUM_CLASSES = len(MY_ANIMALS)

def keep_animals(x_all, y_all, per_class):                 # grab just our animals
    x_out, y_out, counts = [], [], {i: 0 for i in MY_ANIMALS}
    for img, lab in zip(x_all, y_all.flatten()):
        if lab in remap and counts[lab] < per_class:
            x_out.append(img); y_out.append(remap[lab]); counts[lab] += 1
    return np.array(x_out), np.array(y_out)

x_train, y_train = keep_animals(x_train_all, y_train_all, 1500)   # 1500 study photos per animal
x_test,  y_test  = keep_animals(x_test_all,  y_test_all,  400)    # 400 exam photos per animal

print("Today's animals:", class_names)
print("Study photos:", len(x_train), "| Exam photos:", len(x_test))
# 2 --------Generate the animal pictures (small and blurry on purpose)------------
plt.figure(figsize=(13, 5))
for i in range(18):
    plt.subplot(3, 6, i + 1)
    plt.imshow(x_train[i])
    plt.title(class_names[y_train[i]], fontsize=9)
    plt.axis("off")
plt.tight_layout(); plt.show()
# 3 ------Building the brain----------------------
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(32, 32, 3)),              # each photo: 32x32 pixels, 3 colors
    tf.keras.layers.Rescaling(1./255),                     # pixel values 0-255 -> 0-1

    tf.keras.layers.Conv2D(32, 3, activation="relu"),      # 32 pattern detectors
    tf.keras.layers.MaxPooling2D(),                        # shrink, keep strongest signals

    tf.keras.layers.Conv2D(64, 3, activation="relu"),      # 64 detectors, bigger patterns
    tf.keras.layers.MaxPooling2D(),                        # shrink again

    tf.keras.layers.Conv2D(128, 3, activation="relu"),     # 128 detectors, even bigger patterns
    tf.keras.layers.MaxPooling2D(),                        # shrink again

    tf.keras.layers.Flatten(),                             # grid -> one long list of numbers
    tf.keras.layers.Dense(128, activation="relu"),         # combine all the clues
    tf.keras.layers.Dropout(0.3),                          # random forgetting -> less memorizing
    tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),  # SIX scores that add up to 100%
])
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",                # wrongness measure for MANY choices
    metrics=["accuracy"],
)
model.summary()
# 4 --------------------Training the model-----------------------------------
EPOCHS = 50   # one epoch = study every photo once.
history = model.fit(x_train, y_train,
                    validation_data=(x_test, y_test),
                    epochs=EPOCHS)
print(f"\n🎓 Final exam score: {history.history['val_accuracy'][-1]*100:.1f}%")
# 5------------Example of AI's applied thinking-----------------------
preds = model.predict(x_test, verbose=0)        # six scores for every exam photo
plt.figure(figsize=(14, 6))
for i in range(12):
    plt.subplot(2, 6, i + 1)
    plt.imshow(x_test[i])
    guess = class_names[preds[i].argmax()]        # the highest-scoring animal
    truth = class_names[y_test[i]]
    plt.title(f"{guess}\n{preds[i].max()*100:.0f}%", fontsize=9,
              color="green" if guess == truth else "red")
    plt.axis("off")
plt.tight_layout(); plt.show()
# 6-----------Little Game: Human VS. AI-----------------------
confidence = preds.max(axis=1)                  # how sure the AI was about each photo
tricky = confidence.argsort()[:8]               # the 8 it was LEAST sure about

plt.figure(figsize=(14, 4))
for i, idx in enumerate(tricky):
    plt.subplot(1, 8, i + 1)
    plt.imshow(x_test[idx]); plt.axis("off")
    plt.title(f"#{i+1}", fontsize=12)
plt.suptitle("Make your 8 guesses BEFORE closing this window!", fontsize=12)
plt.show()
# now ai's turn:
ai_score = 0
print(" #   AI guessed    sure    truth     AI?")
print("-" * 44)
for i, idx in enumerate(tricky):
    guess = class_names[preds[idx].argmax()]
    truth = class_names[y_test[idx]]
    ok = guess == truth; ai_score += ok
    print(f" {i+1}   {guess:<10}  {preds[idx].max()*100:3.0f}%   {truth:<8}  {'✅' if ok else '❌'}")
print("-" * 44)
print(f"\n🤖 AI got {ai_score}/8 on its HARDEST photos. How many did YOU get?")
# 7------------Data on confusions------------
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
pred_labels = preds.argmax(axis=1)
cm = confusion_matrix(y_test, pred_labels)
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
    ax=ax, cmap="Greens", colorbar=False, xticks_rotation=45)
plt.title("Where does your AI get confused?")
plt.tight_layout(); plt.show()