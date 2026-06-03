# ASCEND Vision System

ASCEND Vision System is the computer vision and perception stack developed for the ISRO Robotics Challenge (IRoC-U). The project focuses on autonomous terrain feature detection, image retrieval, visual localization, mapping, and navigation in GPS-denied environments using a hybrid approach that combines classical computer vision, deep learning embeddings, and SLAM.

The system leverages ORB feature extraction, MobileNetV3-based image embeddings, similarity search, and future ORB-SLAM3 integration to enable reliable feature recognition and autonomous decision-making on resource-constrained edge hardware such as the NVIDIA Jetson Orin Nano.

## Features Implemented

### Preprocessing
- Blur Detection

### Classical Computer Vision
- ORB Feature Extraction
- ORB Feature Matching
- Lowe Ratio Test

### Deep Learning
- MobileNetV3 Feature Extraction
- Embedding Generation
- Cosine Similarity Matching
- Image Retrieval System

## Tech Stack

- Python
- OpenCV
- PyTorch
- TorchVision
- NumPy

## Current Progress

- [x] ORB Feature Pipeline
- [x] MobileNet Embeddings
- [x] Image Retrieval
- [ ] Hybrid Retrieval + ORB Verification
- [ ] ORB-SLAM3 Integration
- [ ] ROS2 Integration
- [ ] Jetson Deployment

## Project Structure

ASCEND-Vision-System
├── preprocessing
├── matching
├── deep_learning
├── dataset
├── results
