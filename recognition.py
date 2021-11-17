from capture_image import capture_image
from detect import *

def image_detect():
    capture_image()

if __name__ == '__main__':
    try:
        image_detect()
        app.run(detect_image)
    except SystemExit:
        pass
