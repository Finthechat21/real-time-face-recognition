# Real-Time Face Recognition

## Overview
This project implements real-time face recognition using OpenCV and the `face_recognition` library. It captures video from the webcam, detects faces, and matches them with preloaded images.

## Features
- Real-time face detection and recognition
- Compares faces with a database of known individuals
- Displays names of recognized faces
- Works with a webcam

## Requirements
Ensure you have the following dependencies installed:

```sh
pip install opencv-python face-recognition numpy
```

## Usage
1. Add images of known individuals in the same directory as the script.
2. Update the `known_images` and `names` lists in the script with corresponding filenames and names.
3. Run the script:

```sh
python project.py
```

4. Press 'q' to exit the program.

## Contributing
Feel free to contribute by improving accuracy, adding new features, or fixing bugs!

## License
This project is open-source and available under the MIT License.

