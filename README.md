# Movie-Genre-Prediction
Predict the genre of any Movie, TV Series, etc using the cover/ poster.

# Concept
The aim that we target is to predict the genre of the movie, right? So, if you give your 
brains a little thinking what we are dealing with is actually a Multi-Label-Classification 
model. Why? A movie can not always belong to a particular genre, like action or comedy. 
The movie can be a combination of two or more genres. Hence, multi-label image classification.
The model that we're building is a specific architecture of Convolutional Neural Networks. Here,
we try to use CNN's to learn and recognize the relationships between a pixel and surrounding pixels 
and try to determine the category/ genre it is most likely to fit into. As we already discussed it
won't just belong to a category so we'll be displaying the top three categories it belongs to along
with their scores. 

# Data and pre-processing
The dataset we’ll be using contains the poster images of several multi-genre movies. 
I have made some changes in the dataset and converted it into a structured format, 
i.e. a folder containing the images and a .csv file for true labels. You can download the structured dataset from [here](https://drive.google.com/file/d/1iQV5kKF_KGZL9ALx9MMXk_Lg7PklBLCE/view?usp=sharing).

# Framework 
For a quick setup and testing I've used TensorFlow. 
Will be adding a PyTorch version anytime soon. 

# GPU and Memory Requirements 
I would highly suggest to work with the model on Cloud as it contains a dataset of 7000 images of size 400*400
and would require high memory(RAM) along with TPUs to speed up the computations. I would suggest a minimum of 
35GB Ram with Google Cloud Engine backend TPUs and nothing less than that, because that's what my model took 
and I had trained it using just 5000 images.

# Final Predictions
The accuracy achieved was 90.86 which surely can be improved with more GPU resources at hand as I had trained the model using only 5000 images out 
of the original 7000 dataset. 

