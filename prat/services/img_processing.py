import numpy as np
import cv2


class ImageProcessing:

    @classmethod
    def box(cls, img:np.array, data:dict) -> np.array:
        """
        Draws a box around the detected object

        Parameters
        ----------
        img: np.array
            Image to draw the box on
        data: dict
            Data from the YOLO service

        Returns
        -------
        np.array
            Image with the box drawn on it
        """
        x1, y1, x2, y2 = int(data["xmin"]), int(data["ymin"]), int(data["xmax"]), int(data["ymax"])

        thickness = int(img.shape[0] * 0.01)
        image = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=thickness)

        return image



# def crop_img(img, data):
#     for i in data:
#         x1 = int(i["xmin"])
#         y1 = int(i["ymin"])
#         x2 = int(i["xmax"])
#         y2 = int(i["ymax"])
#         img_crop = img[y1:y2, x1:x2]
#     return img_crop

    
