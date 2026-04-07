# Video Action Recognition
A simple model to recognise events in a video, specifically designed to detect itching in videos of mice.

## Model Requirements
As this model is intended to be used by scientists using university provided computers, the following criteria were defined:
* Runs inference in realtime or faster on a typical university laptop
 * Ideally without a dedicated gpu
* Easy to install and use
* Provides acceptable accuracy when trained on a limited dataset
* Option to train new models locally or on HPC/cloud infrastructure 
* Additional tools to preprocess videos before inference including automated object tracking

For our specific use-case, the following additional criteria were defined:
* Robust to different mouse strains and appearance
* Minimally affected by camera angle and quality
* Suitable for home cage monitoring with varied backgrounds

## Model Architecture
Traditional mouse activity recognition systems (e.g. B-SOID) use a two-stage system of pose estimation/keypoint tracking to extract video features, followed by a machine learning classifier trained on the extracted features. Three key issues arise from this architecture: 
* error propagation between the two stages harms accuracy
* key visual information (e.g. piloarrection, face scrunching, wincing in mice) is not retained after pose estimation
* pose estimation models require significant datasets to train, and features from the first stage must be manually selected

To address these issues, a fully end-to-end deep learning model was developed, using DINOv2 as a general image feature extractor and a transformer classifier to predict actions from the sequence of image features. The image feature extractor maintains visual information that would be discarded by pose estimation models. Error propagation is reduced by joint training of the two stages and training the classifier on the full set of image features. A pretrained image feature extractor can be applied to a diverse range of targets and can be finetuned on unlabelled datasets, making this suitable for different scenarios, such as live cell imaging. 

We found the features extracted by DINOv2 from videos of mice contain pose information without explicitly training for this task, and DINOv2 features were sufficient to distinguish a range of activities in an unsupervised manner. Supervised training of a transformer classifier on labelled data acheived acceptable accuracy

## Evaluation of the model
This model was evaluated on the MIT mouse dataset and SCRATCH-AID dataset.
### Model size
An essential feature for this model is realtime inference when run on a typical university provided laptop, and therefore reducing model size in important. While the size of DINOv2 is fixed, the number of transformer layers in the classifier can be changed. After evaluation, we found 2 layers was sufficient for acceptable accuracy.
### Inference speed
TODO: evaluate inference speed on a range of hardware
### Dataset size
As labelled datasets are time-consumining to create, this model needs to work well with small datasets to be as useful as possible to researchers. We evaluated the model on a range of training dataset sizes and found a training set of several hundred labelled examples was needed.
### Training speed
TODO: evaluate training speed on a range of hardware
### Robustness to different mouse strains, appearance, background
TODO
### Comparison with other models
TODO