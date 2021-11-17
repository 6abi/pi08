import cv2 

def capture_image():
    key = cv2. waitKey(1)
    webcam = cv2.VideoCapture(0)
    while True:
        try:
            check, frame = webcam.read()
            print(check) #prints true as long as the webcam is running
            print(frame) #prints matrix values of each framecd 
            cv2.imshow("Capturing", frame)
            key = cv2.waitKey(1)
            if key == ord('s'): 
                cv2.imwrite(filename='./data/captured_image.jpg', img=frame)
                webcam.release()
                cv2.waitKey(1650)
                cv2.destroyAllWindows()
                print("Resizing image to 416x416 scale...")
                img_ = cv2.resize(frame,(416,416))
                print("Resized...")
                img_resized = cv2.imwrite(filename='./data/captured_image.jpg', img=img_)
                print("Image saved!")
            
                break
            elif key == ord('q'):
                print("Turning off camera.")
                webcam.release()
                print("Camera off.")
                print("Program ended.")
                cv2.destroyAllWindows()
                break
            
        except(KeyboardInterrupt):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break
