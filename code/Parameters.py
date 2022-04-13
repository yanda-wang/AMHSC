import torch


class Params:
    def __init__(self):
        self.PATIENT_RECORDS_FILE = 'data/records_final.pkl'
        self.PATIENT_RECORDS_FILE_SUPERVISED_HOP = 'data/records_supervised_hop.pkl'
        self.CONCEPTID_FILE = 'data/voc_final.pkl'
        self.EHR_MATRIX_FILE = 'data/ehr_adj_final.pkl'
        self.USE_CUDA = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.USE_CUDA else "cpu")
        self.MEDICATION_COUNT = 153
        self.DIAGNOSES_COUNT = 1960
        self.PROCEDURES_COUNT = 1432

        self.DIAGNOSE_INDEX = 0
        self.PROCEDURE_INDEX = 1
        self.MEDICATION_INDEX = 2
        self.HOP_INDEX_gru_cover = 3

        self.OPT_CALL = 25
        self.OPT_SPLIT_TAG_ADMISSION = -1
        self.OPT_SPLIT_TAG_VARIABLE = -2
        self.OPT_MODEL_MAX_EPOCH = 40

        self.TEACHER_FORCING_RATE = 0.5

        self.LOSS_PROPORTION_BCE = 0.8
        self.LOSS_PROPORTION_Multi_Margin = 0.1
        self.LOSS_PROPORTION_Coverage = 0.1

        self.TRAIN_RATIO = 0.8
        self.TEST_RATIO = 0.1
