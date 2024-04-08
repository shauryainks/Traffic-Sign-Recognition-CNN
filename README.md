**Traffic Sign Recognition using Convolutional Neural Networks**

This project aims to recognize traffic signs using convolutional neural networks (CNNs). It employs the German Traffic Sign Recognition Benchmark (GTSRB) dataset, which consists of images of traffic signs belonging to 43 different categories.

**Project Structure:**
- `traffic.py`: The main Python script containing the code for loading the dataset, building and training the CNN model, and saving the trained model.
- `requirements.txt`: A file listing all the dependencies required to run the project.
- `gtsrb.zip/`: A zip folder containing 43 subfolders, each representing a category of traffic signs. Each subfolder contains images of traffic signs belonging to that category.

**Usage:**
1. Ensure you have Python installed on your system.
2. Install the required dependencies using the following command:
   ```
   pip install -r requirements.txt
   ```
3. Download the GTSRB dataset and extract it. Place the extracted `gtsrb` folder in the same directory as `traffic.py`.
4. Run the `traffic.py` script with the following command:
   ```
   python traffic.py gtsrb
   ```
   Replace `gtsrb` with the path to the directory containing the dataset if it's located elsewhere.
5. Optionally, you can specify a filename to save the trained model:
   ```
   python traffic.py gtsrb model.h5
   ```

By following these steps, you can utilize the provided script to train a convolutional neural network for traffic sign recognition.

**Requirements:**
- Python 3.x
- OpenCV (`cv2`)
- NumPy
- TensorFlow
- scikit-learn

**Model Architecture:**
- Input layer: Accepts images of size 30x30 with 3 color channels (RGB).
- Convolutional layer: Applies 32 filters of size 3x3 with ReLU activation.
- Max-pooling layer: Reduces the spatial dimensions of the image.
- Flatten layer: Converts the 2D output of the convolutional layers into a 1D array.
- Dense hidden layer: Consists of 2048 neurons with ReLU activation and 10% dropout.
- Output layer: Consists of 43 neurons corresponding to the 43 traffic sign categories, with softmax activation.

**Notes:**
- The dataset is split into training and testing sets with a 60:40 ratio.
- The model is trained for 10 epochs with the Adam optimizer and categorical cross-entropy loss.
- The trained model can be saved to a file (optional).

**References:**
- German Traffic Sign Recognition Benchmark (GTSRB) Dataset: http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset
- TensorFlow Documentation: https://www.tensorflow.org/api_docs
- OpenCV Documentation: https://docs.opencv.org/
