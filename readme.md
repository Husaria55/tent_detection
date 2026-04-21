# Tent detection project
### Brainstorm of ideas
1. Use a image-language model that will take as an input a text prompt and an image -> output: is there a tent
2. Use xview dataset, which has a class for tent
3. combine with colors and opencv -- tents are usually bright and it might be possible to do this using some simple algo
4. Use blender for syntethic generation of data

### OpenCV solution
This approach uses only OpenCV operations counting on the fact that tent should differ from the background (colour and shape). 
Unfortunately this approach produces mixed results and is probably unreliable for the comp setup.

### Vision-Language foundation model
First tested model yolov8s-world.pt from ultralytics
Works surprisingly well
![alt text](image.png)