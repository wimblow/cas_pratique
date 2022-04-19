import torch
import logging
import numpy as np
import json



logger = logging.getLogger(__name__)

class YoloService:
    @classmethod
    def __init__(cls):
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        cls.sign_model = torch.hub.load(
            "ultralytics/yolov5", 
            "custom",
            path="prat/models/best_yolov5.pt",
            force_reload=True,
            # source="local",
        )
      
        logger.info("Yolov5 : Model loaded")


    @classmethod
    def yolo_detect(cls, image:np.array) -> dict:
        """
        Detects objects in an image

        Parameters
        ----------
        image: np.array
            Image to detect objects in
        
        Returns
        -------
        dict
            Data from the YOLO service
        """
        listeStorage_yolov5= {}
        data_sign = cls.sign_model(image).pandas().xyxy[0].to_json(orient="records")
        data_sign_dict = json.loads(data_sign)

        return data_sign_dict