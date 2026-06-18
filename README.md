# BTCCAMP — Python & AI Group Projects

A collection of AI/ML models and teaching materials from **BTCCAMP**, a summer camp for refugee students held at James Madison University (JMU). I helped TA the Python & AI group, where we introduced students to machine learning fundamentals using hands-on image classification projects.

---

## About the Camp

BTCCAMP is a local initiative that teaches programming and technology skills to refugee students in the local area. The Python & AI group covered core concepts including python fundamentals, data preprocessing, model training, evaluation, and real-world applications of machine learning using TensorFlow and Keras.

---

## Repository Structure

```
├── cats&dogs_filtered/     # Training data for lesson (day) 2
├── my_photos/# test data/images
├── myprojects/
│   ├── my_tests/           # Training data and lesson images
│   ├── p1/                 # Dangerous vs. safe animal classifier
│   ├── p2/                 # Fish vs. mammal classifier
│   ├── p4/                 # Clothing type classifier (AI stylist)
│   └── p5/                 # Handwritten digit recognizer
├── Day2A.py                # Lesson code for Day 2 partA
├── Day2B.py                # Lesson code for Day 2 partB
├── Day3.py                 # Lesson code for Day 3
├── python_basics.ipynb     # Example lesson notebook (Day 1)
└── requirements.txt
```

---

## Projects

### p1 — Safari (Dangerous) Animal Classifier
A binary image classifier that determines whether an animal is dangerous or safe. Trained on a labeled dataset of animal images using a convolutional neural network (CNN) built with TensorFlow/Keras.

### p2 — Deep Sea (Fish vs. Mammal) Classifier
A binary classifier that distinguishes between fish and mammals from images. Demonstrates fundamental binary classification with CNNs.

### p4 — AI Stylist (Clothing Classifier)
An image classifier that identifies and differentiates between types of clothing. Inspired by the classic Fashion-MNIST problem, adapted for our own dataset.

### p5 — Digit Recognizer
A classifier that recognizes handwritten digits (0–9). A classic introduction to computer vision and multi-class classification.

---

## Teaching Materials

The Jupyter notebooks outside of `myprojects/` are example lesson files used during the camp:

- **Day 1 notebook** — Introductory Python and ML concepts covered in class
- **Day 2 AI ppxt** Introductory to AI and ML concepts covered in class

These were designed to be beginner-friendly for students with little to no prior coding experience.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Primary language |
| TensorFlow / Keras | Model building and training |
| NumPy | Numerical operations |
| OpenCV | Image loading and preprocessing |
| Matplotlib | Visualizing training results |
| scikit-learn | Evaluation metrics |
| Pillow | Image manipulation |
| Jupyter Notebook | Lesson delivery and prototyping |

---

## Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   cd YOUR_REPO_NAME
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac/Linux
   .venv\Scripts\activate         # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Open any notebook**
   ```bash
   jupyter notebook
   ```

---

## Notes

- Models were trained for educational purposes on small datasets — accuracy is intentionally modest to keep training fast and accessible for a camp setting
- p3 is intentionally absent from this repo
- All student-facing materials were designed for beginners with no prior programming experience. Most notebooks are absent for integrity reasons

---

*Built and TA'd at JMU · BTC Summer Camp · Python & AI Group*