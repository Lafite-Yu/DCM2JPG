# from IPython import embed

from DicomFileExtractor import DicomExtractor

import PIL
from PIL import Image
import numpy as np

img = Image.fromarray(np.ones((100, 100)))
img.show('1', 'Gray')

if __name__ == '__main__':
    dicom_path = 'BrainMRI_Sample'
    output_path = 'output_pics'
    dicom_extractor = DicomExtractor(dicom_path, output_path)
    dicom_extractor.run()
    # embed()
