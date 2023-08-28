class BoundingBox:
    cx: float
    cy: float
    width: float
    height: float

    def __init__(self, cx: float, cy: float, width: float, height: float):
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height

class MLData:
    resizing_time: float
    inference_time: float
    masking_time: float
    hash: str
    class_id: str
    confidence: float
    bounding_box: BoundingBox
    name: str

    def __init__(self):
        self.resizing_time = 0.0
        self.inference_time = 0.0
        self.masking_time = 0.0
        self.hash = ''
        self.class_id = ''
        self.confidence = 0.0
        self.bounding_box = None
        self.name = ''

class MLMetadata:
    framekm_data: [MLData]

    def __init__(self):
        self.framekm_data = []
