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
TEAM = 'Team Rummage Realm'
PROJECT = 'AI Stylist'
print(f'🚀 {TEAM} presents: {PROJECT}')
# 1-------------load data--------------
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(xtr, ytr), (xte, yte) = tf.keras.datasets.fashion_mnist.load_data()

class_names = ['T-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
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

x_train, y_train = subsample(xtr, ytr, 2000)
x_test,  y_test  = subsample(xte, yte, 500)
print('Classes:', class_names, '| Study:', len(x_train), '| Exam:', len(x_test))
# 2-----------------check-----------
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
# 3-------------------setting the brain-----------
EPOCHS        = 5
BATCH_SIZE    = 64
LEARNING_RATE = 0.0005
DROPOUT       = 0.30
AUGMENT       = True    # flips are fine for clothes — can be ON or OFF
EXTRA_CONV    = True    # add a deeper detector block. ⬆ accuracy, a bit slower
print('Dials set →', dict(EPOCHS=EPOCHS, BATCH_SIZE=BATCH_SIZE, LEARNING_RATE=LEARNING_RATE, DROPOUT=DROPOUT, AUGMENT=AUGMENT, EXTRA_CONV=EXTRA_CONV))
# 4-------------train the AI----------------------
from tensorflow.keras import layers, models
RUN_NAME = 'run 1: starting point'   # rename each time theres a change on a dial

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
# 5------------check---------------------
cmap = 'gray' if GRAY else None
preds = model.predict(x_test, verbose=0)
plt.figure(figsize=(14, 6))
for i in range(12):
    plt.subplot(2, 6, i + 1); plt.imshow(x_test[i].squeeze(), cmap=cmap)
    guess = class_names[preds[i].argmax()]; truth = class_names[y_test[i]]
    plt.title(f'{guess}\n{preds[i].max()*100:.0f}%', fontsize=9, color='green' if guess == truth else 'red')
    plt.axis('off')
plt.tight_layout(); plt.show()
# 6-----------laundry sorter-----------
TIPS = {'T-shirt': 'casual day', 'trouser': 'gyatt', 'pullover': 'finance pipeline',
        'dress': 'cute', 'coat': 'brrr', 'sandal': 'sumn calm',
        'shirt': 'add a collar', 'sneaker': 'the beaters', 'bag': 'cant forget',
        'ankle boot': 'swifty'}
sample = np.random.choice(len(x_test), 8, replace=False)
pr = model.predict(x_test[sample], verbose=0)
plt.figure(figsize=(15, 4))
for k, idx in enumerate(sample):
    item = class_names[pr[k].argmax()]
    plt.subplot(1, 8, k + 1); plt.imshow(x_test[idx].squeeze(), cmap='gray'); plt.axis('off')
    plt.title(f'{item}\n{TIPS[item]}', fontsize=7)
plt.tight_layout(); plt.show()
# 7-----------confusion matrix----------
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
pred_labels = model.predict(x_test, verbose=0).argmax(axis=1)
cm = confusion_matrix(y_test, pred_labels)
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap='Greens', colorbar=False, xticks_rotation=45)
plt.title('Where does the AI get confused?'); plt.tight_layout(); plt.show()
# 8------------experiment tracker-----------
plt.figure(figsize=(9, max(2, 0.6 * len(experiments))))
names = list(experiments); vals = [experiments[n] for n in names]
plt.barh(names, vals, color='#2C5F2D')
for i, v in enumerate(vals): plt.text(v + 0.5, i, f'{v}%', va='center', fontsize=11)
plt.xlim(0, 100); plt.xlabel('Exam accuracy (%)'); plt.title('Our tuning experiments')
plt.gca().invert_yaxis(); plt.tight_layout(); plt.show()
best = max(experiments, key=experiments.get)
print(f'🥇 Best so far: "{best}" at {experiments[best]}%')
# 9------------real-world test-------------
# Shared helpers for webcam + downloaded photo tests
def snap_photo():
    try:
        import cv2
        import numpy as np
    except ImportError:
        print('OpenCV missing — re-run the Setup cell at the top.'); return None

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print('📷 No webcam found — use the downloaded photo cell instead.'); cap.release(); return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    best_frame = None
    best_sharpness = -1

    for _ in range(70):
        ok, frame = cap.read()
        if not ok:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

        if sharpness > best_sharpness:
            best_sharpness = sharpness
            best_frame = frame.copy()

    cap.release()

    if best_frame is None:
        print('Could not grab a photo — try again.'); return None

    print(f'📷 Photo sharpness: {best_sharpness:.1f}')
    return cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB)


def crop_clothes_part(photo):
    import cv2
    import numpy as np

    img = np.array(photo)
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        center_x = x + w // 2

        top = max(0, y + int(h * 1.15))
        bottom = min(H, y + int(h * 5.0))

        crop_width = int(w * 4.8)
        left = max(0, center_x - crop_width // 2)
        right = min(W, center_x + crop_width // 2)

        print('Face found — cropping mostly clothes area.')

    else:
        left = int(W * 0.08)
        right = int(W * 0.92)
        top = int(H * 0.05)
        bottom = int(H * 0.95)

        print('No face found — using clothing/product crop.')

    clothes = img[top:bottom, left:right]
    return clothes, (left, top, right, bottom)


def make_28x28_versions(clothes_img):
    import cv2
    import numpy as np
    from PIL import Image, ImageOps

    img = np.array(clothes_img.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    versions = {}

    # Version 1: simple invert, good for dark item on light background
    simple = 255 - gray
    simple = Image.fromarray(simple).convert('L')
    simple = ImageOps.autocontrast(simple)
    simple.thumbnail((24, 24))

    canvas = Image.new('L', (28, 28), 0)
    canvas.paste(simple, ((28 - simple.size[0]) // 2, (28 - simple.size[1]) // 2))
    versions['simple_invert'] = np.array(canvas).astype('float32')

    # Version 2 and 3: masked versions
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, mask_dark = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    _, mask_light = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    for mask_name, mask, invert in [
        ('masked_dark_item', mask_dark, True),
        ('masked_light_item', mask_light, False)
    ]:
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean = np.zeros_like(mask)

        if len(contours) > 0:
            biggest = max(contours, key=cv2.contourArea)
            cv2.drawContours(clean, [biggest], -1, 255, thickness=cv2.FILLED)
        else:
            clean = mask

        if invert:
            detail = 255 - gray
        else:
            detail = gray.copy()

        detail[clean == 0] = 0

        ys, xs = np.where(clean > 0)

        if len(xs) > 0 and len(ys) > 0:
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()

            pad = 8
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(detail.shape[1] - 1, x2 + pad)
            y2 = min(detail.shape[0] - 1, y2 + pad)

            detail = detail[y1:y2 + 1, x1:x2 + 1]

        item = Image.fromarray(detail).convert('L')
        item = ImageOps.autocontrast(item)
        item.thumbnail((24, 24))

        canvas = Image.new('L', (28, 28), 0)
        canvas.paste(item, ((28 - item.size[0]) // 2, (28 - item.size[1]) // 2))

        versions[mask_name] = np.array(canvas).astype('float32')

    return versions


def classify_clothing_image(original, crop_mode='auto', expected_group='all'):
    from PIL import Image, ImageDraw
    import numpy as np
    import matplotlib.pyplot as plt

    original = Image.fromarray(original).convert("RGB")
    W, H = original.size

    if crop_mode == "auto":
        clothes, box = crop_clothes_part(np.array(original))
        clothes_img = Image.fromarray(clothes).convert("RGB")

    elif crop_mode == "center":
        side = min(W, H)
        left = (W - side) // 2
        top = (H - side) // 2
        right = left + side
        bottom = top + side
        box = (left, top, right, bottom)
        clothes_img = original.crop(box)

    else:
        box = (0, 0, W, H)
        clothes_img = original.copy()

    boxed = original.copy()
    draw = ImageDraw.Draw(boxed)
    draw.rectangle(box, outline="red", width=4)

    plt.figure(figsize=(6, 4))
    plt.imshow(boxed)
    plt.axis("off")
    plt.title("Original image — red box is used")
    plt.show()

    plt.figure(figsize=(4, 4))
    plt.imshow(clothes_img)
    plt.axis("off")
    plt.title("Image used for prediction")
    plt.show()

    versions = make_28x28_versions(clothes_img)

    if expected_group == "top":
        allowed_names = ["T-shirt", "pullover", "coat", "shirt"]

    elif expected_group == "clothes":
        allowed_names = ["T-shirt", "trouser", "pullover", "dress", "coat", "shirt"]

    else:
        allowed_names = list(class_names)

    allowed_indices = [
        i for i, name in enumerate(class_names)
        if name in allowed_names
    ]

    results = []

    for version_name, arr_raw in versions.items():
        for normalize in [True, False]:

            if normalize:
                arr = arr_raw / 255.0
                norm_name = "div255"
            else:
                arr = arr_raw.copy()
                norm_name = "raw255"

            x = arr[np.newaxis, ..., np.newaxis]
            p = model.predict(x, verbose=0)[0]

            raw_index = p.argmax()
            raw_answer = class_names[raw_index]
            raw_confidence = p[raw_index] * 100

            allowed_scores = p[allowed_indices]
            chosen_position = allowed_scores.argmax()
            chosen_index = allowed_indices[chosen_position]

            filtered_answer = class_names[chosen_index]

            if allowed_scores.sum() > 0:
                filtered_confidence = allowed_scores[chosen_position] / allowed_scores.sum() * 100
            else:
                filtered_confidence = 0

            raw_is_allowed = raw_answer in allowed_names

            results.append({
                "version": version_name,
                "normalize": norm_name,
                "arr": arr,
                "p": p,
                "raw_answer": raw_answer,
                "raw_confidence": raw_confidence,
                "filtered_answer": filtered_answer,
                "filtered_confidence": filtered_confidence,
                "raw_is_allowed": raw_is_allowed
            })

    results_sorted = sorted(
        results,
        key=lambda r: (
            r["raw_is_allowed"],
            r["filtered_confidence"],
            r["raw_confidence"]
        ),
        reverse=True
    )

    best = results_sorted[0]

    if best["raw_is_allowed"]:
        trust = "higher trust"
    else:
        trust = "LOW TRUST — raw model disagrees"

    plt.figure(figsize=(4, 4))
    plt.imshow(best["arr"], cmap="gray")
    plt.axis("off")
    plt.title(
        f"Final guess: {best['filtered_answer']} ({best['filtered_confidence']:.0f}% within {expected_group})\n"
        f"Raw guess: {best['raw_answer']} ({best['raw_confidence']:.0f}%)\n"
        f"{trust}",
        fontsize=11
    )
    plt.show()

    print("Best preprocessing:")
    print(best["version"], "+", best["normalize"])

    print()
    print("Final guess:")
    print(best["filtered_answer"], f"{best['filtered_confidence']:.0f}% within {expected_group}")

    print()
    print("Raw guess:")
    print(best["raw_answer"], f"{best['raw_confidence']:.0f}%")

    print()
    print("All preprocessing attempts:")
    for r in results_sorted:
        print(
            f"{r['version']:18s} {r['normalize']:7s} | "
            f"raw: {r['raw_answer']:10s} {r['raw_confidence']:6.1f}% | "
            f"filtered: {r['filtered_answer']:10s} {r['filtered_confidence']:6.1f}%"
        )

    print()
    print("Top raw guesses for best version:")
    top = np.argsort(best["p"])[::-1][:5]

    for i in top:
        print(f"{class_names[i]}: {best['p'][i] * 100:.1f}%")

print("Shared helpers are ready.")
# --
# 🧪 Snap, crop clothes, make Fashion-MNIST style, and classify

photo = snap_photo()

if photo is not None:
    from PIL import Image, ImageDraw
    import numpy as np
    import matplotlib.pyplot as plt

    clothes, box = crop_clothes_part(photo)

    # Show original photo with crop box
    original = Image.fromarray(photo)
    boxed = original.copy()

    draw = ImageDraw.Draw(boxed)
    draw.rectangle(box, outline='red', width=4)

    plt.figure(figsize=(6, 4))
    plt.imshow(boxed)
    plt.axis('off')
    plt.title('Original photo — red box is the clothes crop')
    plt.show()

    # Show cropped clothes
    clothes_img = Image.fromarray(clothes)

    plt.figure(figsize=(4, 4))
    plt.imshow(clothes_img)
    plt.axis('off')
    plt.title('Cropped clothes part')
    plt.show()

    # Make the image look closer to Fashion-MNIST
    arr = make_fashion_mnist_style(clothes_img)

    x = arr[np.newaxis, ..., np.newaxis]

    p = model.predict(x, verbose=0)[0]

    # =========================
    # CHANGE THIS SETTING
    # =========================
    EXPECTED_GROUP = 'top'  
    # Use:
    # 'top'        for T-shirt / shirt / pullover / coat
    # 'all'        for all Fashion-MNIST classes
    # 'clothes'    for clothes but not shoes/bag
    # =========================

    if EXPECTED_GROUP == 'top':
        allowed_names = ['T-shirt', 'pullover', 'coat', 'shirt']

    elif EXPECTED_GROUP == 'clothes':
        allowed_names = ['T-shirt', 'trouser', 'pullover', 'dress', 'coat', 'shirt']

    else:
        allowed_names = list(class_names)

    allowed_indices = [
        i for i, name in enumerate(class_names)
        if name in allowed_names
    ]

    allowed_scores = p[allowed_indices]
    chosen_position = allowed_scores.argmax()
    chosen_index = allowed_indices[chosen_position]

    answer = class_names[chosen_index]

    if allowed_scores.sum() > 0:
        confidence = allowed_scores[chosen_position] / allowed_scores.sum() * 100
    else:
        confidence = 0

    raw_index = p.argmax()
    raw_answer = class_names[raw_index]
    raw_confidence = p[raw_index] * 100

    # Show what the model actually sees
    plt.figure(figsize=(4, 4))
    plt.imshow(arr, cmap='gray')
    plt.axis('off')

    if EXPECTED_GROUP == 'all':
        plt.title(f'AI thinks: {raw_answer} ({raw_confidence:.0f}%)', fontsize=13)
    else:
        plt.title(
            f'AI thinks: {answer} ({confidence:.0f}% within {EXPECTED_GROUP})\n'
            f'raw top: {raw_answer} ({raw_confidence:.0f}%)',
            fontsize=13
        )

    plt.show()

    print('Top raw guesses:')
    top = np.argsort(p)[::-1][:5]

    for i in top:
        print(f'{class_names[i]}: {p[i] * 100:.1f}%')

    print()
    print('Final prediction:')
    print(answer, f'{confidence:.0f}% within {EXPECTED_GROUP}')

    print()
    print('Class names:')
    print(class_names)

    print()
    print('Important:')
    print('This is still a Fashion-MNIST model, so real webcam photos are hard.')
    print('Best results: hold one clothing item flat, centered, with a plain background.')