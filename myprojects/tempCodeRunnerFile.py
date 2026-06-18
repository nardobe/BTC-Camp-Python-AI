image_folder = "PASTE YOUR FILE PATH HERE"
# image_name = "FILE NAME"

# folder = Path(image_folder)
# files = [folder / image_name]

# files = [f for f in files if f.exists()]

# if not files:
#     print("❌ Image not found.")
#     print("Folder:", folder)
#     print("Image name:", image_name)

# else:
#     imgs = np.array([
#         np.array(Image.open(f).convert("RGB").resize((96, 96)))
#         for f in files
#     ])

#     # IMPORTANT:
#     # If we're training images were divided by 255, uncomment this line:
#     # imgs = imgs / 255.0

#     predictions = model.predict(imgs, verbose=0)

#     n = len(files)
#     plt.figure(figsize=(min(15, 2.8 * n), 3.4))

#     for k, f in enumerate(files):
#         probs = predictions[k]

#         predicted_index = np.argmax(probs)
#         verdict = class_names[predicted_index]
#         confidence = probs[predicted_index]

#         plt.subplot(1, n, k + 1)
#         plt.imshow(Image.open(f).convert("RGB"))
#         plt.axis("off")
#         plt.title(f"{f.name}\n{verdict}\n{confidence * 100:.0f}%", fontsize=10)

#     plt.suptitle("Your AI judging REAL photos from the internet!", fontsize=13)
#     plt.tight_layout()
#     plt.show()

#     print("Raw model prediction:")
#     print(predictions)

#     print("\nClass names:")
#     print(class_names)
