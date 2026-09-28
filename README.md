# AI-Based Facial Recognition Attendance System

October 2025 – December 2025

An attendance system that identifies a student from a photo and can reject a face that is not enrolled. Detection uses MTCNN. Each face is stored as an ArcFace embedding through DeepFace. Identity is decided by cosine distance to the enrolled embeddings.

**Author:** Lara Al Omari

## Problem

Taking attendance by hand is slow and easy to get wrong. This system replaces that step with a face check: a photo is matched to an enrolled student, or marked Unknown when the closest match is too far away.

Each photo filename stores the student name and ID, in the form `StudentName_ID_Number.jpg`. In the interactive tool, a cosine distance above 0.68 is Unknown. The confusion-matrix evaluation always takes the closest enrolled ID, so every test photo receives a prediction.

This is a classification task. The reported results are a confusion matrix, precision, recall, and F1. They are not a regression score.

## Dataset

The photos used for this project are pictures of students. Each person was photographed under different lighting, from different angles, and at different distances, so the matcher is not limited to one studio-style shot.

Those pictures are not in this repository. They are photos of students, and they are left out on purpose.

What is included is the embedding file built from the enrollment photos, `data/student_database.pkl`:

- 57 face embeddings
- 7 students
- ArcFace embeddings of length 512

The original evaluation used 37 held-out photos of those same students and is reported below. To run the project on another class, add your own student photos to the same folders. Do not commit those photos. The code only needs the filename pattern:

`StudentName_ID_Number.jpg`

`StudentName` is the name, `ID` is the student number, and `Number` is which photo of that student it is (`1`, `2`, `3`, …). `.jpeg` and `.png` are also accepted. A trailing photo number is ignored when the name and ID are read, so several pictures of one student stay tied to the same person.

Put the photos here:

- `images/confusionMatrixImages/` for the photos used to build the confusion matrix
- `images/testimages/` for one test-photo set
- `images/test_images/` for a second test-photo set, if you keep that split

Use the same pattern in every folder. The notebook then parses the name and ID and the evaluation runs without a path change inside the matching code.

## Methods

The pipeline builds a facial embedding repository and matches new faces against it.

1. Parse the student name and ID from each filename. A trailing image number is ignored.
2. Detect a face with MTCNN and embed it with ArcFace (512 numbers per face).
3. Save IDs, names, and embeddings in `data/student_database.pkl`.
4. For a new photo, embed the face and compute cosine distance against every enrolled embedding. The nearest neighbor is the candidate identity.
5. The interactive tool accepts that student only when the distance is at most 0.68. A larger distance is classified as Unknown, which is how the system separates enrolled students from unseen faces.
6. The evaluation tool always predicts the closest ID, then builds a confusion matrix and a classification report with scikit-learn. That is the nearest-neighbor check used to score the enrolled set.

## Results

On the 37 photos in `confusionMatrixImages`, forcing the closest ID gave perfect accuracy.

| Student ID | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| 100064694 | 1.00 | 1.00 | 1.00 | 7 |
| 100064785 | 1.00 | 1.00 | 1.00 | 5 |
| 100064788 | 1.00 | 1.00 | 1.00 | 2 |
| 100064840 | 1.00 | 1.00 | 1.00 | 10 |
| 100066678 | 1.00 | 1.00 | 1.00 | 4 |
| 100066739 | 1.00 | 1.00 | 1.00 | 5 |
| 100066740 | 1.00 | 1.00 | 1.00 | 4 |

Overall accuracy is 1.00 on 37 photos of the enrolled students. Those photos differ in lighting, angle, and distance. The plot is `results/confusion_matrix.png`.

The enrollment set and this test set are different photos of the same seven students. The 0.68 threshold is what the identification tool uses when a face is not close enough to anyone enrolled.

## Contribution

Lara Al Omari built the attendance pipeline in `notebooks/AiProjectFinal.ipynb`: the embedding repository, cosine-distance matching with an Unknown threshold, and the confusion-matrix evaluation. The saved ArcFace database is `data/student_database.pkl`.

## How to run

The notebook was written for Google Colab and reads files from Google Drive under `/content/drive/MyDrive/Project AI`. To run it there, put `student_database.pkl` and the photo folders in that Drive folder, then run the notebook from top to bottom.

Locally:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first DeepFace run downloads the ArcFace and MTCNN weights. Point `DATABASE_FILE` in the notebook at `data/student_database.pkl`, and point `TEST_IMAGE_DIR` at `images/confusionMatrixImages`.

Add your own student photos to that folder before the evaluation cell. Name every file `StudentName_ID_Number.jpg`. The original student photos are not provided.

## Repository layout

```
notebooks/AiProjectFinal.ipynb
data/student_database.pkl
results/confusion_matrix.png
images/confusionMatrixImages/
images/testimages/
images/test_images/
requirements.txt
.gitignore
```

## Privacy

The student photos are not published. Anyone reusing the project adds pictures of their own students to `images/confusionMatrixImages/`, `images/testimages/`, and `images/test_images/`, with the filename pattern above.

`data/student_database.pkl` stores face embeddings together with the names and IDs from the original enrollment set. This repository is private. Do not make it public unless everyone in that file has agreed. Image files under `images/` are ignored by git so a local photo folder is not pushed by accident.
