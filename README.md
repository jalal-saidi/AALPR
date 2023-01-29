First Afghan License Plate Reconition System
It contains a pipeline that include:
1) Vehicle Detection
2) License Plate Detection
3) Afghan Characters Segmentation
4) Afghan Characters Recognition

The starter code is from the following project:
https://github.com/sergiomsilva/alpr-unconstrained

The starter code includes vehicle detection, License plate detection, and Latin character recongntion.
The challenge for us was to segment and classify Afghani characters in the detected license plates. We trained two models for Afghani Character Segmentation using Yolo v3 and darknet. For Afghani character classification, we trained a model using Keras. We also optimized the current pipeline using multi threaded queues. For each part of the pipeline, we created a queue in which each frame is processed by different threads. For example, vehicle detection thread, detects and writes vehicles in each frame to its own queue. Meanwhile, license plate detection thread reads and process vehicles from vehicle detection thread. This process continues util the Afghani characters are written to an output video next to each detected license plate.



Inspired by the paper:

@INPROCEEDINGS{silva2018a,
  author={S. M. Silva and C. R. Jung}, 
  booktitle={2018 European Conference on Computer Vision (ECCV)}, 
  title={License Plate Detection and Recognition in Unconstrained Scenarios}, 
  year={2018}, 
  pages={580-596}, 
  doi={10.1007/978-3-030-01258-8_36}, 
  month={Sep},}

