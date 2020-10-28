# object_tracking

Duong Vu

An attempt to build an object tracking with OpenCV and DeepSORT

Object tracking is the process of:

- Taking an initial set of object detections (such as an input set of bounding box coordinates)
- Creating a unique ID for each of the initial detections
- Tracking each of the objects as they move around frames in a video, maintaining the assignment of unique IDs

Object tracking allows us to **apply a unique ID to each tracked object**, making it possible to **count unique objects in a video**. Object tracking is paramount to building a road-users counter (ped, bike, car, truck, etc.).
