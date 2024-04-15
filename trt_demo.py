from utils.utils import BaseEngine
import cv2
import argparse

class Predictor(BaseEngine):
    def __init__(self, engine_path):
        super(Predictor, self).__init__(engine_path)
        self.n_classes = 80  # your model classes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--engine", help="TRT engine Path")
    parser.add_argument("-c", "--conf", type=float, help="confidence threshold")
    parser.add_argument("-i", "--image", help="image path")
    parser.add_argument("-o", "--output", help="image output path")
    parser.add_argument("-v", "--video",  help="video path or camera index ")
    parser.add_argument("--end2end", default=False, action="store_true",
                        help="use end2end engine")

    args = parser.parse_args()
    print(args)

    pred = Predictor(engine_path=args.engine)
    # pred.get_fps()
    img_path = args.image
    video = args.video

    # argument represents a camera index
    if video.isdigit():
       video = int(video)
    conf = args.conf
    if img_path:
      origin_img = pred.inference(img_path, conf=conf, end2end=args.end2end)

      cv2.imwrite("%s" %args.output , origin_img)
    if video:
      pred.detect_video(video, conf=conf, end2end=args.end2end) # set 0 use a webcam
