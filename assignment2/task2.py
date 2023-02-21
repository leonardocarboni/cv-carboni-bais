import numpy as np
import cv2 as cv

background_frames = 20

background_pixels = np.zeros((486, 644, 3), dtype=np.float32)

for camera_i in range(1, 5):
    background_pixels = np.zeros((486, 644, 3), dtype=np.float32)
    with np.load(f'./data/cam{camera_i}/config.npz') as file:
        camera_matrix, dist_coeffs, rvec_extr, tvec_extr, R = [file[i] for i in [
            'camera_matrix', 'dist_coeffs', 'rvec_extr', 'tvec_extr', 'R']]
        cap = cv.VideoCapture(f"./data/cam{camera_i}/background.avi")
        for n_frame in range(background_frames):
            retF, frame = cap.read()
            if retF:
                height, width, channels = frame.shape
                for y in range(0, height):
                    for x in range(0, width):
                        background_pixels[y, x] += frame[y, x]

        cap.release()

        background_pixels = np.array(background_pixels / background_frames, dtype=np.uint8)

        cap = cv.VideoCapture(f"./data/cam{camera_i}/video.avi")
        backSub = cv.bgsegm.createBackgroundSubtractorMOG()
        _ = backSub.apply(background_pixels)
        cv.imshow('mask1', background_pixels)
        cv.waitKey(0)
        cv.destroyAllWindows()
        for n_frame in range(background_frames):
            retF, frame = cap.read()
            if retF:
                _ = backSub.apply(background_pixels)
                mask = backSub.apply(frame)

                mask = cv.morphologyEx(mask, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15)))
                mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))

                # Generate output
                output = cv.bitwise_and(frame, frame, None, mask)
                # foreground = cv.absdiff(frame, background_pixels)
                
                
                # cv.imshow('mask1', foreground)
                # cv.waitKey(0)
                # cv.destroyAllWindows()
                
                # Generate output
                # output = cv.bitwise_and(frame, frame, None, mask1)

                
                

                # cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
                # cv.putText(frame, str(cap.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
                #            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                cv.imshow('mask_morph', mask)
                cv.waitKey(0)
                cv.destroyAllWindows()
                cv.imshow('output', output)
                cv.waitKey(0)
                cv.destroyAllWindows()
                break
