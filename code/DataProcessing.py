import dill
import itertools
import math

import pandas as pd
import numpy as np

from dateutil import relativedelta
from tqdm import tqdm

patients_file = 'data/PATIENTS.csv'
admission_file = 'data/ADMISSIONS.csv'
med_file = 'data/PRESCRIPTIONS.csv'
diag_file = 'data/DIAGNOSES_ICD.csv'
procedure_file = 'data/PROCEDURES_ICD.csv'

ndc2atc_file = 'data/ndc2atc_level4.csv'
cid_atc = 'data/drug-atc.csv'
ndc2rxnorm_file = 'data/ndc2rxnorm_mapping.txt'

PATIENT_RECORDS_FILE = 'data/patient_records.pkl'  # 以ICD和ATC表示疾病以及药品的病人记录，后续会转换成vocabulary表示的记录
PATIENT_RECORDS_FINAL_FILE = 'data/records_final.pkl'

DIAGNOSES_INDEX = 0
PROCEDURES_INDEX = 1
MEDICATIONS_INDEX = 2

VOC_FILE = 'data/voc_final.pkl'
GRAPH_FILE = 'data/ehr_adj_final.pkl'


# ===================处理原始EHR数据，选取对应记录================

def process_procedure():
    pro_pd = pd.read_csv(procedure_file, dtype={'ICD9_CODE': 'category'})
    pro_pd.drop(columns=['ROW_ID'], inplace=True)
    #     pro_pd = pro_pd[pro_pd['SEQ_NUM']<5]
    #     def icd9_tree(x):
    #         if x[0]=='E':
    #             return x[:4]
    #         return x[:3]
    #     pro_pd['ICD9_CODE'] = pro_pd['ICD9_CODE'].map(icd9_tree)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM'], inplace=True)
    pro_pd.drop(columns=['SEQ_NUM'], inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.reset_index(drop=True, inplace=True)

    return pro_pd


def process_med():
    med_pd = pd.read_csv(med_file, dtype={'NDC': 'category'})
    # filter
    med_pd.drop(columns=['ROW_ID', 'DRUG_TYPE', 'DRUG_NAME_POE', 'DRUG_NAME_GENERIC',
                         'FORMULARY_DRUG_CD', 'GSN', 'PROD_STRENGTH', 'DOSE_VAL_RX',
                         'DOSE_UNIT_RX', 'FORM_VAL_DISP', 'FORM_UNIT_DISP', 'FORM_UNIT_DISP',
                         'ROUTE', 'ENDDATE', 'DRUG'], axis=1, inplace=True)
    med_pd.drop(index=med_pd[med_pd['NDC'] == '0'].index, axis=0, inplace=True)
    med_pd.fillna(method='pad', inplace=True)
    med_pd.dropna(inplace=True)
    med_pd.drop_duplicates(inplace=True)
    med_pd['ICUSTAY_ID'] = med_pd['ICUSTAY_ID'].astype('int64')
    med_pd['STARTDATE'] = pd.to_datetime(med_pd['STARTDATE'], format='%Y-%m-%d %H:%M:%S')
    med_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'STARTDATE'], inplace=True)
    med_pd = med_pd.reset_index(drop=True)

    med_pd = med_pd.drop(columns=['ICUSTAY_ID'])
    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)

    # visit > 2
    def process_visit_lg2(med_pd):
        a = med_pd[['SUBJECT_ID', 'HADM_ID']].groupby(by='SUBJECT_ID')['HADM_ID'].unique().reset_index()
        a['HADM_ID_Len'] = a['HADM_ID'].map(lambda x: len(x))
        a = a[a['HADM_ID_Len'] > 1]
        return a

    med_pd_lg2 = process_visit_lg2(med_pd).reset_index(drop=True)
    med_pd = med_pd.merge(med_pd_lg2[['SUBJECT_ID']], on='SUBJECT_ID', how='inner')

    return med_pd.reset_index(drop=True)


def process_diag():
    diag_pd = pd.read_csv(diag_file)
    diag_pd.dropna(inplace=True)
    diag_pd.drop(columns=['SEQ_NUM', 'ROW_ID'], inplace=True)
    diag_pd.drop_duplicates(inplace=True)
    diag_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID'], inplace=True)
    return diag_pd.reset_index(drop=True)


def ndc2atc4(med_pd):
    with open(ndc2rxnorm_file, 'r') as f:
        ndc2rxnorm = eval(f.read())
    med_pd['RXCUI'] = med_pd['NDC'].map(ndc2rxnorm)
    med_pd.dropna(inplace=True)

    rxnorm2atc = pd.read_csv(ndc2atc_file)
    rxnorm2atc = rxnorm2atc.drop(columns=['YEAR', 'MONTH', 'NDC'])
    rxnorm2atc.drop_duplicates(subset=['RXCUI'], inplace=True)
    med_pd.drop(index=med_pd[med_pd['RXCUI'].isin([''])].index, axis=0, inplace=True)

    med_pd['RXCUI'] = med_pd['RXCUI'].astype('int64')
    med_pd = med_pd.reset_index(drop=True)
    med_pd = med_pd.merge(rxnorm2atc, on=['RXCUI'])
    med_pd.drop(columns=['NDC', 'RXCUI'], inplace=True)
    med_pd = med_pd.rename(columns={'ATC4': 'NDC'})
    med_pd['NDC'] = med_pd['NDC'].map(lambda x: x[:4])
    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)
    return med_pd


def filter_1000_most_pro(pro_pd):
    pro_count = pro_pd.groupby(by=['ICD9_CODE']).size().reset_index().rename(columns={0: 'count'}).sort_values(
        by=['count'], ascending=False).reset_index(drop=True)
    pro_pd = pro_pd[pro_pd['ICD9_CODE'].isin(pro_count.loc[:1000, 'ICD9_CODE'])]

    return pro_pd.reset_index(drop=True)


def filter_2000_most_diag(diag_pd):
    diag_count = diag_pd.groupby(by=['ICD9_CODE']).size().reset_index().rename(columns={0: 'count'}).sort_values(
        by=['count'], ascending=False).reset_index(drop=True)
    diag_pd = diag_pd[diag_pd['ICD9_CODE'].isin(diag_count.loc[:1999, 'ICD9_CODE'])]

    return diag_pd.reset_index(drop=True)


def filter_300_most_med(med_pd):
    med_count = med_pd.groupby(by=['NDC']).size().reset_index().rename(columns={0: 'count'}).sort_values(by=['count'],
                                                                                                         ascending=False).reset_index(
        drop=True)
    med_pd = med_pd[med_pd['NDC'].isin(med_count.loc[:299, 'NDC'])]

    return med_pd.reset_index(drop=True)


def process_ehr():
    # get med and diag (visit>=2)
    med_pd = process_med()
    med_pd = ndc2atc4(med_pd)
    #     med_pd = filter_300_most_med(med_pd)

    diag_pd = process_diag()
    diag_pd = filter_2000_most_diag(diag_pd)

    pro_pd = process_procedure()
    #     pro_pd = filter_1000_most_pro(pro_pd)

    med_pd_key = med_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    diag_pd_key = diag_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    pro_pd_key = pro_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()

    combined_key = med_pd_key.merge(diag_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    combined_key = combined_key.merge(pro_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

    diag_pd = diag_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    med_pd = med_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    pro_pd = pro_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

    # flatten and merge
    diag_pd = diag_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])['ICD9_CODE'].unique().reset_index()
    med_pd = med_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])['NDC'].unique().reset_index()
    pro_pd = pro_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])['ICD9_CODE'].unique().reset_index().rename(
        columns={'ICD9_CODE': 'PRO_CODE'})
    med_pd['NDC'] = med_pd['NDC'].map(lambda x: list(x))
    pro_pd['PRO_CODE'] = pro_pd['PRO_CODE'].map(lambda x: list(x))
    data = diag_pd.merge(med_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    data = data.merge(pro_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    #     data['ICD9_CODE_Len'] = data['ICD9_CODE'].map(lambda x: len(x))
    data['NDC_Len'] = data['NDC'].map(lambda x: len(x))

    patient_records = []
    for subject_id in data['SUBJECT_ID'].unique():
        item_df = data[data['SUBJECT_ID'] == subject_id]
        patient = []
        for index, row in item_df.iterrows():
            admission = []
            admission.append([item for item in row['ICD9_CODE']])  # diagnoses
            admission.append([item for item in row['PRO_CODE']])  # procedures
            admission.append([item for item in row['NDC']])  # medications
            patient.append(admission)

        patient_records.append(patient)

    dill.dump(patient_records, open(PATIENT_RECORDS_FILE, 'wb'))


# ==================构建字典，将原始EHR记录转换为medical code==============
class Voc(object):
    def __init__(self):
        self.idx2word = {0: 'CLS', 1: 'SEP'}
        self.word2idx = {'CLS': 0, 'SEP': 1}

    def add_sentence(self, sentence):
        for code in sentence:
            if code not in self.word2idx:
                self.idx2word[len(self.word2idx)] = code
                self.word2idx[code] = len(self.word2idx)

    def get_token_count(self):
        return len(self.idx2word)


def build_voc():
    diag_voc = Voc()
    pro_voc = Voc()
    med_voc = Voc()
    patient_records = dill.load(open(PATIENT_RECORDS_FILE, 'rb'))

    for patient in patient_records:
        for adm in patient:
            diagnoses, procedures, medications = adm[DIAGNOSES_INDEX], adm[PROCEDURES_INDEX], adm[MEDICATIONS_INDEX]
            diag_voc.add_sentence(diagnoses)
            pro_voc.add_sentence(procedures)
            med_voc.add_sentence(medications)

    dill.dump({'diag_voc': diag_voc, 'pro_voc': pro_voc, 'med_voc': med_voc}, open(VOC_FILE, 'wb'))


def convert_patient_records():
    voc = dill.load(open(VOC_FILE, 'rb'))
    diag_voc = voc['diag_voc']
    pro_voc = voc['pro_voc']
    med_voc = voc['med_voc']
    patient_records = dill.load(open(PATIENT_RECORDS_FILE, 'rb'))
    patient_records_idx = []

    for patient in patient_records:
        current_patient = []
        for adm in patient:
            diagnoses, procedures, medications = adm[DIAGNOSES_INDEX], adm[PROCEDURES_INDEX], adm[MEDICATIONS_INDEX]
            admission = []
            admission.append([diag_voc.word2idx[item] for item in diagnoses])
            admission.append([pro_voc.word2idx[item] for item in procedures])
            admission.append([med_voc.word2idx[item] for item in medications])
            current_patient.append(admission)
        patient_records_idx.append(current_patient)

    dill.dump(patient_records_idx, open(PATIENT_RECORDS_FINAL_FILE, 'wb'))


def build_graph():
    voc = dill.load(open(VOC_FILE, 'rb'))
    medication_count = len(voc['med_voc'].word2idx)
    ehr_matrix = np.zeros((medication_count, medication_count))
    patient_records = pd.read_pickle(PATIENT_RECORDS_FINAL_FILE)
    for patient in tqdm(patient_records):
        for adm in patient:
            medications = adm[MEDICATIONS_INDEX]
            for (med_1, med_2) in itertools.combinations(medications, 2):
                ehr_matrix[med_1, med_2] = 1
                ehr_matrix[med_2, med_1] = 1

    dill.dump(ehr_matrix, open(GRAPH_FILE, 'wb'))


if __name__ == '__main__':
    process_ehr()
    build_voc()
    convert_patient_records()
    build_graph()
