#from jetson_inference import detectNet
#from jetson_utils import videoSource, videoOutput
import jetson.inference
import jetson.utils

image_paths = [
    "image1.jpg",
    "image2.jpg"
    ]

thresholds = [0.3, 0.7]

net = jetson.inference.detectNet(
    "ssd-mobilenet-v2",
    threshold=thresholds[0],
)

for image_path in image_paths:
    for threshold in thresholds:
        net.SetThreshold(threshold)
        # load image
        img = jetson.utils.loadImage(image_path)        
        print("current image:", image_path)
        print("current threshold:", threshold)
        print("image size:", img.width, "x" , img.height)

        # detect
        detections = net.Detect(img)
        print("Detected objects:", len(detections))
        for detection in detections:
            print("ClassID:", detection.ClassID,
                 "Confidence:", detection.Confidence,
                 "Center:", detection.Center,
                 "Width:", detection.Width,
                 "Height:", detection.Height,
                 "Left&Top:", detection.Left, detection.Top,
                 "Right&Bottom:", detection.Right, detection.Bottom,
                 "Area:", detection.Area,
            )
        output_path = image_path.rsplit(".", 1)[0] + f"_threshold_{threshold}.jpg"
        jetson.utils.saveImage(output_path, img)


