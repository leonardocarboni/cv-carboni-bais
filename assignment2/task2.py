import numpy as np
import cv2 as cv

num_frames = 20

background_pixels = np.zeros((486, 644, 3), dtype=np.float32)

for camera_i in range(1, 5):
    background_pixels = np.zeros((486, 644, 3), dtype=np.float32)
    with np.load(f'./data/cam{camera_i}/config.npz') as file:
        camera_matrix, dist_coeffs, rvec_extr, tvec_extr, R = [file[i] for i in [
            'camera_matrix', 'dist_coeffs', 'rvec_extr', 'tvec_extr', 'R']]
        cap = cv.VideoCapture(f"./data/cam{camera_i}/background.avi")
        for n_frame in range(num_frames):
            retF, frame = cap.read()
            if retF:
                height, width, channels = frame.shape
                for y in range(0, height):
                    for x in range(0, width):
                        background_pixels[y, x] += frame[y, x]

        cap.release()

        background_pixels = np.array(background_pixels / num_frames, dtype=np.uint8)

        cap = cv.VideoCapture(f"./data/cam{camera_i}/video.avi")
        #backSub = cv.bgsegm.createBackgroundSubtractorMOG()
        # backSub2 = cv.createBackgroundSubtractorMOG2()
        # print(f"shadow:{backSub2.getShadowThreshold()}")
        #backSub2.setShadowThreshold(1)
        # cv.imshow('mask1', background_pixels)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        for n_frame in range(num_frames):
            retF, frame = cap.read()
            if retF:
                #foreground = cv.absdiff(frame, background_pixels)
                frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                w, h, _ = frame.shape
                mask = np.zeros((w, h), dtype = np.uint8)
                background_pixels_hsv = cv.cvtColor(background_pixels, cv.COLOR_BGR2HSV)
                foreground_hsv = cv.absdiff(frame_hsv, background_pixels_hsv)
                thresh = 30
                # cv.imshow('foreground_gray', foreground_hsv)
                # cv.waitKey(0)
                # cv.destroyAllWindows()
                #foreground_hsv = cv.inRange(foreground_hsv, (0, 0, thresh), (180, 100, 100))
                # _, mask = cv.threshold(foreground_gray,thresh,255,cv.THRESH_BINARY)
                for x in range(foreground_hsv.shape[0]):
                    for y in range(foreground_hsv.shape[1]):
                        if  foreground_hsv[x, y, 0] >= 5 and foreground_hsv[x, y, 1] >= 0 and foreground_hsv[x, y, 2] >= 0:
                            mask[x, y] = 255
                # # foreground = cv.cvtColor(foreground_hsv, cv.COLOR_GRAY2BGR)
                
                # mask = cv.morphologyEx(mask, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
                # mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
                # #mask = cv.morphologyEx(mask, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10)))
                contours, hierarchy = cv.findContours(image=mask, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
                frame_copy = frame.copy()
                cv.drawContours(image=frame_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)
                cv.imshow('contours', frame_copy)
                cv.waitKey(0)
                cv.destroyAllWindows()
                output = cv.bitwise_and(frame, frame, mask=mask)
                cv.imshow('mask', mask)
                cv.waitKey(0)
                cv.destroyAllWindows()
                cv.imshow('output', output)
                cv.waitKey(0)
                cv.destroyAllWindows()
                np.savez(f"data/cam{camera_i}/mask", mask=mask)
                break
                # #_ = backSub.apply(background_pixels)
                # _ = backSub2.apply(background_pixels)

                # #mask = backSub.apply(frame)
                # mask2 = backSub2.apply(frame)

                # #mask = cv.morphologyEx(mask, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (1, 1)))
                # #mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (55, 55)))
                
                

                # mask2 = cv.morphologyEx(mask2, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (8, 8)))
                # mask2 = cv.morphologyEx(mask2, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (40, 40)))
                # mask2 = cv.morphologyEx(mask2, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (25, 25)))

                # # kernel = np.ones((10,10),np.uint8)
                # # mask2 = cv.erode(mask2,kernel,iterations = 1)
                # # # Generate output
                # #output = cv.bitwise_and(frame, frame, None, mask)
                # #output = cv.cvtColor(output, cv.COLOR_HSV2BGR)
                # output2 = cv.bitwise_and(frame, frame, None, mask2)
                # output2 = cv.cvtColor(output2, cv.COLOR_HSV2BGR)
                # # # foreground = cv.absdiff(frame, background_pixels)
                
                
                # # # cv.imshow('mask1', foreground)
                # # # cv.waitKey(0)
                # # # cv.destroyAllWindows()
                
                # # # Generate output
                # # # output = cv.bitwise_and(frame, frame, None, mask1)

                
                

                # # # cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
                # # # cv.putText(frame, str(cap.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
                # # #            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                # #cv.imshow('mask_morph', mask)
                # cv.imshow('mask_morph2', mask2)
                # cv.waitKey(0)
                # cv.destroyAllWindows()
                # #cv.imshow('output', output)
                # cv.imshow('output2', output2)
                # cv.waitKey(0)
                # cv.destroyAllWindows()
                # break
