# CNN-Animal-Classification

A project to classify 10 animal species using a Convolutional Neural Network (CNN) model. The project includes data preprocessing, augmentation, and accuracy comparison under different conditions.

## Project Objective

This project aims to develop and evaluate a CNN model for classifying images of animals. The model's performance is analyzed under different conditions, including brightness manipulation and color constancy application.

## Technologies Used

- **Python Libraries**:
  - TensorFlow
  - NumPy
  - OpenCV
  - Matplotlib
  - Scikit-learn
- **Platform**:
  - Google Colab

## Steps

1. **Data Preparation**:
   - Filtering and balancing the dataset for 10 animal classes.
   - Resizing and normalizing images to 128x128 dimensions.

2. **Data Augmentation**:
   - Techniques applied include:
     - Brightness adjustment
     - Rotation, zoom, and shifts
     - Gaussian noise addition
     - Contrast enhancement using CLAHE

3. **Model Training**:
   - A CNN model with multiple Conv2D, MaxPooling, BatchNormalization layers, and Dense layers.
   - Early stopping used to avoid overfitting.

4. **Evaluation**:
   - Comparison of model performance on:
     - Original test set
     - Manipulated test set (e.g., brightness altered)
     - Color constancy-adjusted test set

5. **Results Visualization**:
   - Training and validation accuracy and loss graphs.
   - Comparison of test accuracies across conditions.

## Dataset

- The dataset is sourced from Kaggle: [Animals with Attributes 2](https://www.kaggle.com/datasets/rrebirrth/animals-with-attributes-2)
- 10 animal classes with 650 images each:
  - Collie, Dolphin, Elephant, Fox, Moose, Rabbit, Sheep, Squirrel, Giant Panda, Polar Bear

## Results

- Test accuracy on the original dataset: **61.44%**
- Test accuracy on manipulated dataset: **11.18%**
- Test accuracy after color constancy adjustment: **11.18%**

## Setup and Usage

### Dependencies

Install the required dependencies:
```bash
pip install tensorflow opencv-python matplotlib scikit-learn
```

### Running the Notebook

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/CNN-Animal-Classification.git
   ```
2. Navigate to the project directory:
   ```bash
   cd CNN-Animal-Classification
   ```
3. Open and run the Jupyter Notebook:
   ```bash
   jupyter notebook notebook.ipynb
   ```

## File Structure

- `README.md`: Project documentation
- `notebook.ipynb`: The complete project code
- `images/`: Result graphs and visuals
- `data/`: Processed datasets
- `models/`: Saved model files

## Visualizations

The project includes graphs showing:
- Training vs. validation accuracy
- Training vs. validation loss

## Future Improvements

- Incorporate additional data augmentation techniques.
- Optimize the model architecture for better performance.
- Experiment with advanced preprocessing methods like histogram equalization or channel shifts.

## Acknowledgments

This project is based on the "Animals with Attributes 2" dataset, sourced from Kaggle.
