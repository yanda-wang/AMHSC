import torch
import dill
import os
import pickle
import sys

import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pylab as p

from torch.autograd import Variable
from pandas import Series
from tqdm import tqdm
from collections import Counter
from itertools import combinations
from sklearn.metrics import average_precision_score

from Networks import EncoderLinearQuery, EncoderLinearAdap
from Networks import DecoderGRUCoverFixed, DecoderGRUCoverAdapLastQuery

from Parameters import Params

params = Params()


class EvaluationUtil:
    def precision_auc(self, predict_prob, target_prescriptions):
        return average_precision_score(target_prescriptions, predict_prob, average='macro')

    def metric_jaccard_similarity(self, predict_prescriptions, target_prescriptions):
        union = list(set(predict_prescriptions) | set(target_prescriptions))
        intersection = list(set(predict_prescriptions) & set(target_prescriptions))
        jaccard = float(len(intersection)) / len(union)
        return jaccard

    def metric_precision(self, predict_prescriptions, target_prescriptions):
        if len(set(predict_prescriptions)) == 0:
            return 0
        intersection = list(set(predict_prescriptions) & set(target_prescriptions))
        # precision = float(len(intersection)) / len(set(predict_prescriptions))
        precision = float(len(intersection)) / len(predict_prescriptions)
        return precision

    def metric_recall(self, predict_prescriptions, target_prescriptions):
        intersection = list(set(predict_prescriptions) & set(target_prescriptions))
        # recall = float(len(intersection)) / len(set(target_prescriptions))
        recall = float(len(intersection)) / len(target_prescriptions)
        return recall

    def metric_f1(self, precision, recall):
        if precision + recall == 0:
            return 0
        f1 = 2.0 * precision * recall / (precision + recall)
        return f1


class FixedHopEvaluation:
    def __init__(self, device, voc_file, patient_records_file, predict_prob_thershold=0.5, ehr_matrix_file=None):
        self.device = device
        self.patient_records_file = patient_records_file
        self.voc_file = voc_file
        self.predict_prob_thershold = predict_prob_thershold
        self.ehr_matrix_file = ehr_matrix_file

        voc = dill.load(open(self.voc_file, 'rb'))
        self.diag_voc = voc['diag_voc']
        self.pro_voc = voc['pro_voc']
        self.med_voc = voc['med_voc']

        self.diagnose_count = len(self.diag_voc.word2idx)
        self.procedure_count = len(self.pro_voc.word2idx)
        self.medication_count = len(self.med_voc.word2idx)

        self.ehr_matrix = dill.load(open(self.ehr_matrix_file, 'rb'))
        self.evaluate_utils = EvaluationUtil()

    def metric_jaccard_similarity(self, predict_medications, target_medications):
        return self.evaluate_utils.metric_jaccard_similarity(predict_medications, target_medications)

    def metric_precision(self, predict_medications, target_medications):
        return self.evaluate_utils.metric_precision(predict_medications, target_medications)

    def metric_recall(self, predict_medications, target_medications):
        return self.evaluate_utils.metric_recall(predict_medications, target_medications)

    def metric_f1(self, precision, recall):
        return self.evaluate_utils.metric_f1(precision, recall)

    def metric_prauc(self, predict_prob, target_multi_hot):
        return self.evaluate_utils.precision_auc(predict_prob, target_multi_hot)

    def evaluateIters(self, encoder, decoder, decoder_type, patient_records, save_result_file=None):
        total_jaccard, total_precision, total_recall, total_f1, total_prauc = [], [], [], [], []
        predict_result_patient_records = []
        for i, patient in enumerate(tqdm(patient_records)):
            current_patient = []
            for idx, adm in enumerate(patient):
                current_records = patient[:idx + 1]

                query, memory_keys, memory_values = encoder(current_records)
                predict_output = decoder(query, memory_keys, memory_values)

                target_medications = adm[params.MEDICATION_INDEX]
                target_multi_hot = np.zeros(self.medication_count)
                target_multi_hot[target_medications] = 1
                predict_prob = torch.sigmoid(predict_output).detach().cpu().numpy()[0]
                predict_multi_hot = predict_prob.copy()

                index_nan = np.argwhere(np.isnan(predict_multi_hot))
                if index_nan.shape[0] != 0:
                    predict_multi_hot = np.zeros_like(predict_multi_hot)

                predict_multi_hot[predict_multi_hot >= self.predict_prob_thershold] = 1
                predict_multi_hot[predict_multi_hot < self.predict_prob_thershold] = 0
                predict_medications = list(np.where(predict_multi_hot == 1)[0])

                jaccard = self.metric_jaccard_similarity(predict_medications, target_medications)
                precision = self.metric_precision(predict_medications, target_medications)
                recall = self.metric_recall(predict_medications, target_medications)
                f1 = self.metric_f1(precision, recall)
                prauc = self.metric_prauc(predict_prob, target_multi_hot)

                total_jaccard.append(jaccard)
                total_precision.append(precision)
                total_recall.append(recall)
                total_f1.append(f1)
                total_prauc.append(prauc)

                adm.append(predict_medications)
                current_patient.append(adm)

            predict_result_patient_records.append(current_patient)

        jaccard_avg = np.mean(np.array(total_jaccard))
        precision_avg = np.mean(np.array(total_precision))
        recall_avg = np.mean(np.array(total_recall))
        f1_avg = np.mean(np.array(total_f1))
        prauc_avg = np.mean(np.array(total_prauc))

        dill.dump(obj=predict_result_patient_records,
                  file=open(os.path.join(save_result_file, 'predict_result.pkl'), 'wb'))

        print('evaluation result:')
        print('  jaccard:', jaccard_avg)
        print('precision:', precision_avg)
        print('   recall:', recall_avg)
        print('       f1:', f1_avg)
        print('    prauc:', prauc_avg)

    def evaluate(self, load_model_name, input_size, hidden_size, encoder_n_layers, encoder_embedding_dropout_rate,
                 encoder_gru_dropout_rate, decoder_dropout_rate, decoder_hop_count, regular_hop_count, attn_type_kv,
                 attn_type_embedding, least_adm_count, select_adm_count, coverage_dim, save_result_path=None):

        print('load model from checkpoint file:', load_model_name)
        checkpoint = torch.load(load_model_name, map_location='cpu')

        encoder = EncoderLinearQuery(self.device, input_size, hidden_size, self.diagnose_count,
                                     self.procedure_count, encoder_n_layers, encoder_embedding_dropout_rate,
                                     encoder_gru_dropout_rate)
        decoder = DecoderGRUCoverFixed(params.device, hidden_size, self.medication_count, decoder_dropout_rate,
                                       least_adm_count, decoder_hop_count, coverage_dim, attn_type_kv,
                                       attn_type_embedding, regular_hop_count, self.ehr_matrix)

        encoder_sd = checkpoint['encoder']
        decoder_sd = checkpoint['decoder']
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
        encoder = encoder.to(self.device)
        decoder = decoder.to(self.device)
        encoder.eval()
        decoder.eval()

        print('load patient records >>>')

        patient_records = pd.read_pickle(self.patient_records_file)
        split_point = int(len(patient_records) * params.TRAIN_RATIO)
        test_count = int(len(patient_records) * params.TEST_RATIO)
        patient_records_train = patient_records[:split_point]
        patient_records_test = patient_records[split_point:split_point + test_count]
        patient_records_validation = patient_records[split_point + test_count:]

        if not os.path.exists(save_result_path):
            os.makedirs(save_result_path)

        print('start evaluation >>>')
        self.evaluateIters(encoder, decoder, patient_records_validation, save_result_path)


class AdaphopLastQueryEvaluation:
    def __init__(self, device, voc_file, patient_records_file, predict_prob_thershold=0.5, ehr_matrix_file=None):
        self.device = device
        self.patient_records_file = patient_records_file
        self.voc_file = voc_file
        self.predict_prob_thershold = predict_prob_thershold
        self.ehr_matrix_file = ehr_matrix_file

        voc = dill.load(open(self.voc_file, 'rb'))
        self.diag_voc = voc['diag_voc']
        self.pro_voc = voc['pro_voc']
        self.med_voc = voc['med_voc']

        self.diagnose_count = len(self.diag_voc.word2idx)
        self.procedure_count = len(self.pro_voc.word2idx)
        self.medication_count = len(self.med_voc.word2idx)

        self.ehr_matrix = dill.load(open(self.ehr_matrix_file, 'rb'))
        self.evaluate_utils = EvaluationUtil()

    def metric_jaccard_similarity(self, predict_medications, target_medications):
        return self.evaluate_utils.metric_jaccard_similarity(predict_medications, target_medications)

    def metric_precision(self, predict_medications, target_medications):
        return self.evaluate_utils.metric_precision(predict_medications, target_medications)

    def metric_recall(self, predict_medications, target_medications):
        return self.evaluate_utils.metric_recall(predict_medications, target_medications)

    def metric_f1(self, precision, recall):
        return self.evaluate_utils.metric_f1(precision, recall)

    def metric_prauc(self, predict_prob, target_multi_hot):
        return self.evaluate_utils.precision_auc(predict_prob, target_multi_hot)

    def evaluateIters(self, encoder, decoder, patient_records, save_result_file=None):
        total_jaccard, total_precision, total_recall, total_f1, total_prauc = [], [], [], [], []
        predict_result_patient_records = []
        target_hop_tag_gru_cover = {}
        predict_hop_tag = {}
        predict_hop_tag[-3] = 0
        for i, patient in enumerate(tqdm(patient_records)):
            current_patient = []
            for idx, adm in enumerate(patient):
                current_records = patient[:idx + 1]
                target_medications = adm[params.MEDICATION_INDEX]
                target_multi_hot = np.zeros(self.medication_count)
                target_multi_hot[target_medications] = 1
                query, memory_keys, memory_values, hop_tag_gru_cover = encoder(current_records)
                if hop_tag_gru_cover in target_hop_tag_gru_cover.keys():
                    target_hop_tag_gru_cover[hop_tag_gru_cover] = target_hop_tag_gru_cover[hop_tag_gru_cover] + 1
                else:
                    target_hop_tag_gru_cover[hop_tag_gru_cover] = 1

                predict_output, hop_output = decoder(query, memory_keys, memory_values)

                if hop_output is not None:
                    predict_hop_probability = torch.sigmoid(hop_output)
                    topv, topi = predict_hop_probability.topk(1)
                    predict_hop_count = topi[0][0].detach().cpu().numpy().item()

                    if predict_hop_count in predict_hop_tag.keys():
                        predict_hop_tag[predict_hop_count] = predict_hop_tag[predict_hop_count] + 1
                    else:
                        predict_hop_tag[predict_hop_count] = 1

                predict_medication_prob = torch.sigmoid(predict_output).detach().cpu().numpy()[0]
                predict_multi_hot = predict_medication_prob.copy()
                index_nan = np.argwhere(np.isnan(predict_multi_hot))
                if index_nan.shape[0] != 0:
                    predict_multi_hot = np.zeros_like(predict_multi_hot)
                predict_multi_hot[predict_multi_hot >= self.predict_prob_thershold] = 1
                predict_multi_hot[predict_multi_hot < self.predict_prob_thershold] = 0
                predict_medications = list(np.where(predict_multi_hot == 1)[0])

                jaccard = self.metric_jaccard_similarity(predict_medications, target_medications)
                precision = self.metric_precision(predict_medications, target_medications)
                recall = self.metric_recall(predict_medications, target_medications)
                f1 = self.metric_f1(precision, recall)
                prauc = self.metric_prauc(predict_medication_prob, target_multi_hot)

                total_jaccard.append(jaccard)
                total_precision.append(precision)
                total_recall.append(recall)
                total_f1.append(f1)
                total_prauc.append(prauc)

                adm.append(predict_medications)
                current_patient.append(adm)

            predict_result_patient_records.append(current_patient)

        jaccard_avg = np.mean(np.array(total_jaccard))
        precision_avg = np.mean(np.array(total_precision))
        recall_avg = np.mean(np.array(total_recall))
        f1_avg = np.mean(np.array(total_f1))
        prauc_avg = np.mean(np.array(total_prauc))

        dill.dump(obj=predict_result_patient_records,
                  file=open(os.path.join(save_result_file, 'predict_result.pkl'), 'wb'))

        print('evaluation result:')
        print('  jaccard:', jaccard_avg)
        print('precision:', precision_avg)
        print('   recall:', recall_avg)
        print('       f1:', f1_avg)
        print('    prauc:', prauc_avg)

        # print('ground truth hop count')
        # for key in sorted(target_hop_tag_gru_cover.keys()):
        #     print(key, target_hop_tag_gru_cover[key])
        #
        # print('predict hop count')
        # for key in sorted(predict_hop_tag.keys()):
        #     print(key, predict_hop_tag[key])

    def evaluate(self, load_model_name, input_size, hidden_size, encoder_n_layers, encoder_embedding_dropout_rate,
                 encoder_gru_dropout_rate, decoder_dropout_rate, decoder_max_hop_count, regular_hop_count,
                 attn_type_kv, attn_type_embedding, least_adm_count, coverage_dim, save_result_path=None):
        print('load model from checkpoint file:', load_model_name)
        checkpoint = torch.load(load_model_name, map_location=torch.device('cpu'))

        encoder = EncoderLinearAdap(self.device, input_size, hidden_size, self.diagnose_count, self.procedure_count,
                                    encoder_n_layers, encoder_embedding_dropout_rate, encoder_gru_dropout_rate)

        decoder = DecoderGRUCoverAdapLastQuery(self.device, hidden_size, self.medication_count,
                                               decoder_dropout_rate, least_adm_count, decoder_max_hop_count,
                                               coverage_dim, attn_type_kv, attn_type_embedding, regular_hop_count,
                                               self.ehr_matrix)

        encoder_sd = checkpoint['encoder']
        decoder_sd = checkpoint['decoder']
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
        encoder = encoder.to(self.device)
        decoder = decoder.to(self.device)
        encoder.eval()
        decoder.eval()

        print('load patient records >>>')

        patient_records = pd.read_pickle(self.patient_records_file)
        patient_records_test = patient_records['test']
        patient_records_validation = patient_records['validation']

        if not os.path.exists(save_result_path):
            os.makedirs(save_result_path)

        print('start evaluation >>>')
        self.evaluateIters(encoder, decoder, patient_records_validation, 'data/model')
