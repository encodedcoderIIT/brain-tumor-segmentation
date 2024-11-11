# Brain Tumor Segmentation

## Description

This project focuses on the segmentation of brain tumors from MRI images using deep learning techniques. The goal is to accurately identify and segment tumor regions to assist in medical diagnosis and treatment planning.

## Installation

### Prerequisites

- Python 3.8+
- Git

### Steps

1. Clone the repository:
   ```sh
   git clone https://github.com/encodedcoderIIT/brain-tumor-segmentation.git
   ```
2. Navigate to the project directory:
   ```sh
   cd brain-tumor-segmentation
   ```
3. Create a virtual environment:
   ```sh
   conda create --prefix ./brats-web-app python=3.8
   ```
4. Activate the virtual environment:

   ```sh
   # On Windows
   conda activate ./brats-web-app

   # On macOS/Linux
   source venv/bin/activate
   ```

5. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Dependencies

- TensorFlow 2.4.1
- Keras 2.4.3
- NumPy 1.19.5
- OpenCV 4.5.1
- Scikit-learn 0.24.1
- Matplotlib 3.3.4
- nibabel 5.2.1

## Usage

## Project Structure

- `data/`: Directory containing the dataset.
- `models/`: Directory to save trained models.
- `train.py`: Script to train the segmentation model.
- `evaluate.py`: Script to evaluate the trained model.
- `segment.py`: Script to segment new MRI images.
- `utils.py`: Utility functions for data preprocessing and visualization.

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

## License

- WIP
