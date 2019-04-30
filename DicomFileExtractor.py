import csv
import os

import numpy as np
import pydicom as dicom
from PIL import Image

from TypeExtractor import TypeExtractor

OUTPUT_FORMAT_JPG = True  # 'JPG' or 'PNG'
IGNORE_EXISTED_EXAMS = False
IGNORE_EXISTED_RESULTS = False
OUTPUT_IMAGE = False
OUTPUT_CSV = False
OUTPUT_SERIES_DESCRIPTION_TYPE = True


class DicomExtractor:
    def __init__(self, in_dicom_path, in_output_path):
        self.dicom_path = in_dicom_path
        self.filename_extension = '.jpg' if OUTPUT_FORMAT_JPG else '.png'
        self.output_path = in_output_path
        self.type_extractor = TypeExtractor()

    def run(self):
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

        self.patients_extractor()
        if OUTPUT_SERIES_DESCRIPTION_TYPE:
            self.type_extractor.generate_res()

    # sub folders of './BrainMRI'
    def patients_extractor(self):
        if not os.path.exists(os.path.join(self.output_path, self.dicom_path)):
            os.mkdir(os.path.join(self.output_path, self.dicom_path))

        patients_names = os.listdir(self.dicom_path)
        for patient_iter in patients_names:
            if not os.path.isdir(os.path.join(self.dicom_path, patient_iter)):
                continue
            print("Extracting patient name: {}".format(patient_iter))
            self.type_extractor.add_patient(patient_iter)
            per_patient_path = os.path.join(self.dicom_path, patient_iter)
            self.exams_extractor(per_patient_path)

    # sub folders of './BrainMRI/Somebody'
    def exams_extractor(self, patient_path):
        if not os.path.exists(os.path.join(self.output_path, patient_path)):
            os.mkdir(os.path.join(self.output_path, patient_path))

        exams_names = os.listdir(patient_path)
        for exam_iter in exams_names:
            per_exam_path = os.path.join(patient_path, exam_iter)
            if exam_iter != 'Viewer' and os.path.isdir(per_exam_path):
                if IGNORE_EXISTED_EXAMS:
                    if os.path.exists(os.path.join(self.output_path, per_exam_path)):
                        print("\tExam existed, ignored: {}.".format(per_exam_path))
                        continue
                if OUTPUT_IMAGE or OUTPUT_CSV:
                    print('\tExtracting exam name: {}'.format(exam_iter))
                success_count = self.exam_extractor(per_exam_path)
                if OUTPUT_IMAGE or OUTPUT_CSV:
                    print('\tFinished. {} succeed'.format(success_count))

    # sub folders of './BrainMRI/Somebody/20180203001894'
    def exam_extractor(self, exam_path):
        if not os.path.exists(os.path.join(self.output_path, exam_path)):
            os.mkdir(os.path.join(self.output_path, exam_path))

        cases_name = os.listdir(exam_path)
        success_count = 0
        for case_iter in cases_name:
            per_case_path = os.path.join(exam_path, case_iter)
            if os.path.isdir(per_case_path):
                if OUTPUT_IMAGE or OUTPUT_CSV:
                    print('\t\tExtracting case name: {}'.format(case_iter))
                per_success_count = self.case_extractor(exam_path, case_path_name=case_iter)
                if OUTPUT_IMAGE or OUTPUT_CSV:
                    print('\t\tFinished. {} succeed'.format(per_success_count))
                    success_count += per_success_count
        return success_count

    # files in in './BrainMRI/刘国旺/20180223001894/1_5455D123B29241C28B797B81ACE98940'
    def case_extractor(self, exam_path, case_path_name):
        case_path = os.path.join(exam_path, case_path_name)
        file_names = os.listdir(case_path)

        success_count = 0
        # self.type_extractor.add_images(file_names, case_path)
        res, store_path = self.type_extractor.add_images_v2(file_names, case_path)
        if not res:
            return 0
        store_path = os.path.join(exam_path, store_path)

        if not os.path.exists(os.path.join(self.output_path, store_path)):
            os.mkdir(os.path.join(self.output_path, store_path))
        if not os.path.exists(os.path.join(self.output_path, store_path, 'csv_files')):
            os.mkdir(os.path.join(self.output_path, store_path, 'csv_files'))

        if OUTPUT_IMAGE or OUTPUT_CSV:
            for filename_iter in file_names:
                if IGNORE_EXISTED_RESULTS:
                    csv_file_path = os.path.join(self.output_path, store_path, 'csv_files',
                                                 filename_iter.replace('.dcm', '.csv'))
                    if os.path.exists(csv_file_path):
                        continue
                self.file_extractor(case_path, filename_iter, store_path)
                success_count += 1
        return success_count

    def file_extractor(self, case_path, filename, store_path):
        dcm_file_path = os.path.join(case_path, filename)
        pic_file_path = os.path.join(self.output_path, store_path,
                                     filename.replace('.dcm', self.filename_extension))
        csv_file_path = os.path.join(self.output_path, store_path, 'csv_files',
                                     filename.replace('.dcm', '.csv'))

        with open(csv_file_path, 'w', newline='') as csv_file:
            try:
                ds = dicom.dcmread(dcm_file_path)
            except:
                print(filename)
                return False

            fieldnames = ds.dir()
            # self.type_extractor.add_type(ds.get('SeriesDescription'))

            if OUTPUT_IMAGE:
                try:
                    pixel_array_numpy = ds.pixel_array
                    pixel_array_numpy = np.array(pixel_array_numpy, dtype=np.uint16) / np.max(pixel_array_numpy) * 255
                    pixel_array_numpy = np.array(pixel_array_numpy, dtype=np.uint8)
                except AttributeError:
                    print('[WARNING] No Pixel Data, only csv file will be generated: {}.'
                          .format(filename))
                else:
                    dcm_image = Image.fromarray(pixel_array_numpy)
                    dcm_image = dcm_image.convert('RGB')
                    dcm_image.save(pic_file_path)

            if OUTPUT_CSV:
                writer = csv.writer(csv_file, delimiter=',')
                writer.writerow(fieldnames)
                rows = []
                for field in fieldnames:
                    if field != 'PixelData':
                        try:
                            if ds.data_element(field) is None:
                                rows.append('')
                            else:
                                x = ds.get(field)
                                rows.append(x)
                        except KeyError:
                            rows.append('')
                    else:
                        rows.append('PIXEL DATA')
                writer.writerow(rows)
