
# Audio Classification of Infant Cries

This project aims to classify baby cry based on the acoustic signals along with demographic features of babies using machine learning techniques. Acoustic features (MFCC, Chroma features, and other spectral features) and Mel Spectrograms are extracted from the audio files from the Donate-A-Cry dataset. We employed Random Forest, SVM, Vision Transformer, and CNN for cry reason classification.

Baby Cry Classification involves pattern recognition to differentiate cry signals, and the process includes audio processing, feature extraction, feature selection, and classification techniques to analyze and categorize baby cries. 

This research aims to classify baby cry based on the acoustic cry signals and Mel spectrogram extracted from audio files. We are interested in training models with a combination of these acoustic features, such as MFCC, Spectral and Chroma features, and comparing the performance of different machine learning methods on baby cry classification.




## Dataset
We have used the Donate-A-Cry corpus dataset, which consists of 457 children cry audio files along with tags for babies’ demographic information (age and gender) and cry reasons.

Acoustic Features
Mel Frequency Cepstral Coefficients (MFCCs) are a set of features derived from the log short-term power spectrum of a sound signal on a nonlinear mel scale of frequency.
## Machine Learning Algorithms and Techniques
The project will use four classification algorithms: Random Forest, Transformer, Support Vector Machine, and Convolution Neural Network. 

For Random Forest and Support Vector Machine, inputs are acoustic features extracted from audio waves of baby cries, such as Mel-frequency cepstral coefficients and chroma features, and demographic features of babies, including age and gender. 

For Transformer, the inputs are the Mel-Spectrogram images extracted from audio files. Outputs are labels of cry reasons: belly pain, burping, discomfort, hunger, and tiredness. 

For CNN, we extracted Mel-Spectrogram images and MFCC features from the audio files as input to the model.

For the class imbalance problem, two techniques were selected to tackle the issue, which are: class weights and SMOTE. Setting the class weight to ‘balanced’ automatically adjusts the weights by giving more importance to the minority class samples during training whereas SMOTE is used to oversample the minority class by generating synthetic samples to balance the class distribution in the data.
## Results and Conclusions
-Utilizing SMOTE in Random Forest classifier slightly increases the model performance in the minority group 

-Data augmentation with random zooming and normalization makes the model learn spurious pattern in the data and adding class weight doesn’t improve model performance for Transformer classifier. 

-In SVM, the model with full length audio files showed better performance than the 1 second audio file model and also the class weight technique proved to be slightly better than SMOTE to handle imbalanced data

-For CNN,  the best results were derived from audio length of 1 second using MFCC values and with dropout layers added to the architecture for dealing with overfitting

-Comparing all the models, Random Forest gives us slightly better weighted F1 score. We finally achieved an F1 score of 76% with accuracy of 84% by implementing various techniques like Data augmentation, SMOTE, Dropout layers to improve model performance. 
