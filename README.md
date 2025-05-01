# Digital Image Processing - Assignment #03
This digital image processing assignment covers a wide range of image processing techniques such as thresholding, contour detection, morphological operations, and object classification.

## 📁 Folder Structure

```
.
├── data/
│   ├── circle3.jpg
│   ├── circles.jpg
│   ├── thermal.jpg
│   ├── text.jpg
│   ├── soduku.jpg
│   ├── alphabets-urdu.jpg
│   └── bubble.jpg
├── main.py
└── README.md
```

> ⚠️ Make sure the `data/` directory contains the required input images as shown above.

---

## 🔧 Requirements

- Python 3.x
- OpenCV (cv2)
- NumPy

Install dependencies using pip:

```bash
pip install opencv-python numpy
```

---

## 🚀 How to Run

Run any of the individual tasks by calling the corresponding function at the end of `main.py`:

```python
# Uncomment the desired function to run
q1()
q2()
q3()
q4()
q5()
q6()
```

Then run the script:

```bash
python main.py
```

---

## 🧠 Task Descriptions

### `q1()` – Circle Image Analysis

- **(a)** Detects and draws centroids of contours.
- **(b)** Identifies red centroids and dilates them to fill holes.
- **(c)** Computes diameter of the largest contour's hole.
- **(d)** Distinguishes between objects with and without holes.

### `q2()` – Thermal Image Person Detection

- Detects small high-temperature regions in a thermal image.
- Draws bounding boxes around possible humans based on area thresholding.

### `q3()` – Text Image Analysis

- **(a)** Counts the number of text lines.
- **(b)** Counts the number of words using size and shape filtering.

### `q4()` – Sudoku Grid Analysis

- Detects empty squares (cells with no black pixels) in a Sudoku grid.
- Fills those cells with red for visualization.

### `q5()` – Urdu Alphabets Image Analysis

- **(a)** Highlights characters with width > height.
- **(b)** Highlights characters with width ≈ height.
- **(c)** Detects and marks "nuktas" (small dots).
- **(e)** Detects holes inside alphabet contours.

### `q6()` – Bubble Sheet Reader

- Splits a bubble sheet into a grid and detects the selected bubble (A-D) per row based on black pixel counts.

---

## 📝 Notes

- Ensure image files are named and stored correctly in the `data/` folder.
- Use raw strings (`r'path'`) or forward slashes (`'path/to/image.jpg'`) for cross-platform compatibility.
