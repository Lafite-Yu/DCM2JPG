import csv
import os

import numpy as np
# import cv2
import pydicom as dicom
from PIL import Image


class TypeExtractor:
    def __init__(self):

        # self.predefined_types = {
        #     'T1': {'Ax': {'AX T1 FLAIR',
        #                   'Ax T1,Flair',
        #                   'OAx T1 FLAIR'},
        #            'Sag': {'Sag T1,Flair',
        #                    'Sag T1 Flair',
        #                    'OSag T1 FLAIR'},
        #            'Cor': {'PosDisp: [14] T1 TIR cor P3'}},
        #     'T1_C': {'Ax': {'Ax T1,Flair +C',
        #                     'AX T1+C',
        #                     'OAx T1+C',
        #                     't1_tir_tra_P3',
        #                     't1_tirm_tra_dark-fluid'},
        #              'Sag': {'Sag CUBE T1 +C',
        #                      'T1 TIR Sagittal P3',
        #                      't1_tirm_sag_dark-fluid',
        #                      'Sag T1,Flair +C',
        #                      'SAG T1+C',
        #                      'OSag T1+C'},
        #              'Cor': {'PosDisp: [10] T1 TIR cor P3',
        #                      'PosDisp: [11] T1 TIR cor P3',
        #                      'T1 TIR cor P3',
        #                      't1_tirm_cor_dark-fluid',
        #                      'Cor T1,Flair +C',
        #                      'COR T1+C',
        #                      'OCor T1+C'}},
        #     'T2': {'Ax': {'AX T2 FRFSE',
        #                   'Ax T2 FSE',
        #                   'FL:C/OAx T2 PROPELLER',
        #                   't2_tse_tra_320_p2',
        #                   't2_tse_tra_P2_24slice'},
        #            'Others': {'3-Pl T2* FGRE'}},
        #     'T2_Flair': {'Ax': {'Ax Flair irFSE',
        #                         't2_FLAIR_P3',
        #                         't2_tirm_tra_dark-fluid',
        #                         'OAx T2 FLAIR',
        #                         'OAx T2 FLAIR+C'}}
        # }
        self.predefined_types = {
            'AX T1 FLAIR': 'T1_Ax',
            'Ax T1,Flair': 'T1_Ax',
            'OAx T1 FLAIR': 'T1_Ax',
            'Sag T1,Flair': 'T1_Sag',
            'OSag T1 FLAIR': 'T1_Sag',
            'Sag T1 Flair': 'T1_Sag',
            'PosDisp: [14] T1 TIR cor P3': 'T1_Cor',
            'OAx T1+C': 'T1_C_Ax',
            'Ax T1,Flair +C': 'T1_C_Ax',
            'AX T1+C': 'T1_C_Ax',
            't1_tirm_tra_dark-fluid': 'T1_C_Ax',
            't1_tir_tra_P3': 'T1_C_Ax',
            't1_tirm_sag_dark-fluid': 'T1_C_Sag',
            'OSag T1+C': 'T1_C_Sag',
            'T1 TIR Sagittal P3': 'T1_C_Sag',
            'SAG T1+C': 'T1_C_Sag',
            'Sag CUBE T1 +C': 'T1_C_Sag',
            'Sag T1,Flair +C': 'T1_C_Sag',
            'COR T1+C': 'T1_C_Cor',
            'T1 TIR cor P3': 'T1_C_Cor',
            'PosDisp: [10] T1 TIR cor P3': 'T1_C_Cor',
            't1_tirm_cor_dark-fluid': 'T1_C_Cor',
            'OCor T1+C': 'T1_C_Cor',
            'Cor T1,Flair +C': 'T1_C_Cor',
            'PosDisp: [11] T1 TIR cor P3': 'T1_C_Cor',
            'AX T2 FRFSE': 'T2_Ax',
            'Ax T2 FSE': 'T2_Ax',
            't2_tse_tra_320_p2': 'T2_Ax',
            'FL:C/OAx T2 PROPELLER': 'T2_Ax',
            't2_tse_tra_P2_24slice': 'T2_Ax',
            '3-Pl T2* FGRE': 'T2_Others',
            't2_FLAIR_P3': 'T2_Flair_Ax',
            't2_tirm_tra_dark-fluid': 'T2_Flair_Ax',
            'Ax Flair irFSE': 'T2_Flair_Ax',
            'OAx T2 FLAIR+C': 'T2_Flair_Ax',
            'OAx T2 FLAIR': 'T2_Flair_Ax'
        }

        self.patient_types = {}
        self.patient_count_per_type_once = {}
        self.patient_count_per_type_image_nums = {}
        self.last_patient = ''
        self.all_types = set()
        self.type_list = ['T1_Ax', 'T1_Sag', 'T1_Cor',
                          'T1_C_Ax', 'T1_C_Sag', 'T1_C_Cor',
                          'T2_Ax', 'T2_Others',
                          'T2_Flair_Ax',
                          'Screen Save', 'Exponential Apparent Diffusion Coefficient',
                          'Apparent Diffusion Coefficient (mm2|s)', 'OAx DWI Asset',
                          'OAx T2 PROPELLER ', 'ep2d_diff_3scan_trace_p2',
                          'ep2d_diff_3scan_trace_p2_ADC', 'localizer', 'PhoenixZIPReport',
                          'PosDisp: [3] t2_tse_tra_P2_24slice ', 'PosDisp: [8] T1 TIR cor P3 ',
                          'PosDisp: [9] T1 TIR cor P3 ', 'Mag_Images', 'csi_se_30_fast',
                          'mIP_Images(SW)', 'Pha_Images', 'SWI_Images',
                          'FILT_PHA: OAx SWI By Zhang Yingkui',
                          'MPR Ob_Ax_I -> S_Min IP_sp:5.0_th:5.0', 'OAx SWI By Zhang Yingkui',
                          'MultiPlanar Reconstruction (MPR) Ob_Ax_S -> I_Min IP_sp:5.0_th:5',
                          'FL:C|OAx T2 PROPELLER', 'Processed Images', 'OAx PWI 50 Phase',
                          'ASSET Calibration', 'SCREENSAVE', 'Ax DWI 1000b', 'LOC', '2D CSI-PROBE 35',
                          'OAx eDWI b=10', '2D CSI-PROBE 144', 'CEST_ssfse', 'Ax DTI 13 Directions',
                          'AX 3D ASL', 'TOF_3D_multi-slab_MIP_SAG', 'TOF_3D_multi-slab_MIP_COR',
                          'TOF_3D_multi-slab', '<MIP Range[1]>', 'TOF_3D_multi-slab_MIP_TRA',
                          '<MIP Range>']

    def add_patient(self, patient):
        self.patient_types[patient] = set()
        self.patient_count_per_type_once[patient] = 49 * [0]
        self.patient_count_per_type_image_nums[patient] = 49 * ['']
        self.last_patient = patient

    def add_images_v2(self, image_names, case_path):
        image_nums = len(image_names)

        dcm_file_path = os.path.join(case_path, image_names[0])
        mri_type = dicom.dcmread(dcm_file_path).get('SeriesDescription').replace('/', '|')
        self.patient_types[self.last_patient].add(mri_type)
        self.all_types.add(mri_type)

        if mri_type in self.predefined_types:
            mri_type = self.predefined_types[mri_type]
            # print(mri_type)
        mri_type_index = self.type_list.index(mri_type)
        self.patient_count_per_type_once[self.last_patient][mri_type_index] += 1
        self.patient_count_per_type_image_nums[self.last_patient][mri_type_index] += '_{}'.format(image_nums)
        return True, '{}_{}'.format(mri_type, self.patient_count_per_type_once[self.last_patient][mri_type_index])

    def add_images(self, image_names, case_path):
        image_nums = len(image_names)
        n = int(image_nums / 2)
        filename = image_names[n]

        dcm_file_path = os.path.join(case_path, filename)
        ds = dicom.dcmread(dcm_file_path)

        mri_type = ds.get('SeriesDescription').replace('/', '|')
        self.patient_types[self.last_patient].add(mri_type)
        self.all_types.add(mri_type)

        if not os.path.exists('series_description_samples'):
            os.mkdir('series_description_samples')
        if not os.path.exists(os.path.join('series_description_samples', mri_type)):
            os.mkdir(os.path.join('series_description_samples', mri_type))

        self.write_img(ds, filename, mri_type, dcm_file_path)
        self.write_csv(ds, filename, mri_type)

    def write_img(self, ds, filename, mri_type, dcm_file_path):
        try:
            pixel_array_numpy = ds.pixel_array
            pixel_array_numpy = np.array(pixel_array_numpy, dtype=np.uint16) / np.max(pixel_array_numpy) * 255
            pixel_array_numpy = np.array(pixel_array_numpy, dtype=np.uint8)
        except AttributeError:
            print('[WARNING] No Pixel Data: {}.'.format(filename))
        else:
            pic_file_path = os.path.join('series_description_samples', mri_type,
                                         self.last_patient + '_' + filename.replace('.dcm', '.jpg'))
            dcm_image = Image.fromarray(pixel_array_numpy)
            dcm_image = dcm_image.convert('RGB')
            dcm_image.save(pic_file_path)

    def write_csv(self, ds, filename, mri_type):
        csv_file_path = os.path.join('series_description_samples', mri_type,
                                     self.last_patient + '_' + filename.replace('.dcm', '.csv'))
        with open(csv_file_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            fieldnames = ds.dir()
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

    def generate_res(self):
        for k, v in self.patient_types.items():
            print(k)
            for item in v:
                print('{{{}}}'.format(item), end="\t")
            print()

        self.csv_generate()
        # print(self.type_list)

    def csv_generate(self):
        all_types = list(self.all_types)
        patients = self.patient_types
        with open('series_description.csv', 'w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(['Name'] + all_types)
            for name, type_set in patients.items():
                rows = [name]
                for item in all_types:
                    if item in type_set:
                        rows.append(1)
                    else:
                        rows.append(0)
                writer.writerow(rows)

        with open('SD_per_image_once.csv', 'w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(['Name'] + self.type_list)
            for name, type_set in self.patient_count_per_type_once.items():
                rows = [name]
                for item in type_set:
                    rows.append(item)
                writer.writerow(rows)

        with open('SD_per_image_numsOfImages.csv', 'w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(['Name'] + self.type_list)
            for name, type_set in self.patient_count_per_type_image_nums.items():
                rows = [name]
                for item in type_set:
                    rows.append(item)
                writer.writerow(rows)
