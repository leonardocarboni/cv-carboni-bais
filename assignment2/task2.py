import numpy as np
import cv2 as cv

num_frames = 20

background_pixels = np.zeros((486, 644, 3), dtype=np.float32)

# known values for best threshold after search, just to make execution go faster
best_masks = {'1': (2, 0, 15), '2': (10, 0, 30), '3': (4, 0, 30), '4': (1, 0, 25)}

def find_best_mask(ground_truth, foreground_hsv):
    best_differences = ground_truth.shape[0] * ground_truth.shape[1]
    best_mask = np.zeros((w, h), dtype = np.uint8)
    for hue in range(0,11):
        for saturation in range(0, 6, 5):
            for value in range(0, 31, 5):
                mask = np.zeros((w, h), dtype = np.uint8)
                for x in range(foreground_hsv.shape[0]):
                    for y in range(foreground_hsv.shape[1]):
                        if  foreground_hsv[x, y, 0] > hue and foreground_hsv[x, y, 1] > saturation and foreground_hsv[x, y, 2] > value:
                            mask[x, y] = 255
                differences = np.sum(cv.bitwise_xor(ground_truth, mask)) / 255
                if differences < best_differences:
                    print(hue, saturation, value)
                    best_mask = mask
                    best_differences = differences
    print(best_differences)
    return best_mask

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
        ground_truth = cv.imread(f"./data/cam{camera_i}/ground_truth.jpg", cv.IMREAD_GRAYSCALE)
        for n_frame in range(num_frames):
            retF, frame = cap.read()
            if retF:
                frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                w, h, _ = frame.shape
                background_pixels_hsv = cv.cvtColor(background_pixels, cv.COLOR_BGR2HSV)
                foreground_hsv = cv.absdiff(frame_hsv, background_pixels_hsv)
                
                # call to perform search TODO: uncomment
                # best_mask = find_best_mask(ground_truth, foreground_hsv)
                
                # call to use known value, uncomment to go faster
                best_mask = np.zeros((w, h), dtype = np.uint8)
                hue, saturation, value = best_masks[str(camera_i)]
                for x in range(foreground_hsv.shape[0]):
                    for y in range(foreground_hsv.shape[1]):
                        if  foreground_hsv[x, y, 0] > hue and foreground_hsv[x, y, 1] > saturation and foreground_hsv[x, y, 2] > value:
                            best_mask[x, y] = 255
                

                cv.imshow('contours', best_mask)
                cv.waitKey(0)
                cv.destroyAllWindows()
                best_mask = cv.morphologyEx(best_mask, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2)))
                best_mask = cv.morphologyEx(best_mask, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15)))
                #mask = cv.morphologyEx(mask, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10)))

                output = cv.bitwise_and(frame, frame, mask=best_mask)
                cv.imshow('mask', best_mask)
                cv.waitKey(0)
                cv.destroyAllWindows()
                cv.imshow('output', output)
                cv.waitKey(0)
                cv.destroyAllWindows()
                np.savez(f"data/cam{camera_i}/mask", mask=best_mask)
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
