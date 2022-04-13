import argparse

import torch
import os
import sys
import datetime
import pickle
import dill
import random

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import optim
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from random import choices
from scipy.stats import entropy, boxcox
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from Networks import EncoderLinearQuery, EncoderLinearAdap
from Networks import DecoderGRUCoverFixed, DecoderGRUCoverAdapLastQuery, DecoderGRUCoverFixedCheckTruth

from Parameters import Params

params = Params()
GRU_COVER_MAX_HOP = 14


class FixedHopTraining:
    def __init__(self, device, patient_records_file, voc_file, ehr_matrix_file):
        self.device = device
        self.patient_records_file = patient_records_file
        self.voc_file = voc_file
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

    def loss_function(self, target_medications, predict_medications, proportion_bce, proportion_multi):
        loss_bce_target = np.zeros((1, self.medication_count))
        loss_bce_target[:, target_medications] = 1
        loss_multi_target = np.full((1, self.medication_count), -1)
        for idx, item in enumerate(target_medications):
            loss_multi_target[0][idx] = item

        loss_bce = F.binary_cross_entropy_with_logits(predict_medications,
                                                      torch.FloatTensor(loss_bce_target).to(self.device))
        loss_multi = F.multilabel_margin_loss(torch.sigmoid(predict_medications),
                                              torch.LongTensor(loss_multi_target).to(self.device))
        loss = proportion_bce * loss_bce + proportion_multi * loss_multi

        return loss

    def get_performance_on_testset(self, encoder, decoder, patient_records):
        jaccard_avg, precision_avg, recall_avg, f1_avg, prauc_avg = [], [], [], [], []
        count = 0
        for patient in patient_records:
            for idx, adm in enumerate(patient):
                count += 1
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

                predict_multi_hot[predict_multi_hot >= 0.5] = 1
                predict_multi_hot[predict_multi_hot < 0.5] = 0
                predict_medications = list(np.where(predict_multi_hot == 1)[0])

                jaccard = self.evaluate_utils.metric_jaccard_similarity(predict_medications, target_medications)
                precision = self.evaluate_utils.metric_precision(predict_medications, target_medications)
                recall = self.evaluate_utils.metric_recall(predict_medications, target_medications)
                f1 = self.evaluate_utils.metric_f1(precision, recall)
                prauc = self.evaluate_utils.precision_auc(predict_prob, target_multi_hot)

                jaccard_avg.append(jaccard)
                precision_avg.append(precision)
                recall_avg.append(recall)
                f1_avg.append(f1)
                prauc_avg.append(prauc)

        jaccard_avg = np.mean(np.array(jaccard_avg))
        precision_avg = np.mean(np.array(precision_avg))
        recall_avg = np.mean(np.array(recall_avg))
        f1_avg = np.mean(np.array(f1_avg))
        prauc_avg = np.mean(np.array(prauc_avg))
        # loss_avg = np.mean(np.array(loss_avg))

        return jaccard_avg, precision_avg, recall_avg, f1_avg, prauc_avg

    def trainIters(self, encoder, decoder, encoder_optimizer, decoder_optimizer, patient_records_train,
                   patient_records_test, save_model_path, n_epoch, loss_proportion_bce, loss_proportion_multi,
                   print_every_iteration=100, save_every_epoch=5, trained_epoch=0, trained_iteration=0):
        start_epoch = trained_epoch + 1
        trained_n_iteration = trained_iteration
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)
        log_file = open(os.path.join(save_model_path, 'medrec_loss.log'), 'a+')
        encoder_lr_scheduler = ReduceLROnPlateau(encoder_optimizer, mode='max', patience=5, factor=0.1)
        decoder_lr_scheduler = ReduceLROnPlateau(decoder_optimizer, mode='max', patience=5, factor=0.1)

        for epoch in range(start_epoch, start_epoch + n_epoch):
            print_loss = []
            iteration = 0
            for patient in patient_records_train:
                for idx, adm in enumerate(patient):
                    trained_n_iteration += 1
                    iteration += 1
                    current_records = patient[:idx + 1]
                    target_medications = adm[params.MEDICATION_INDEX]
                    encoder_optimizer.zero_grad()
                    decoder_optimizer.zero_grad()

                    query, memory_keys, memory_values = encoder(current_records)
                    predict_output = decoder(query, memory_keys, memory_values)
                    loss = self.loss_function(target_medications, predict_output, loss_proportion_bce,
                                              loss_proportion_multi)
                    print_loss.append(loss.item())
                    loss.backward()
                    encoder_optimizer.step()
                    decoder_optimizer.step()

                    if iteration % print_every_iteration == 0:
                        print_loss_avg = np.mean(np.array(print_loss))
                        print_loss = []
                        print(
                            'epoch: {}; time: {}; Iteration: {}; train loss: {:.4f}'.format(
                                epoch, datetime.datetime.now(), trained_n_iteration, print_loss_avg))
                        log_file.write(
                            'epoch: {}; time: {}; Iteration: {};  train loss: {:.4f}\n'.format(
                                epoch, datetime.datetime.now(), trained_n_iteration, print_loss_avg))

            encoder.eval()
            decoder.eval()
            jaccard_avg, precision_avg, recall_avg, f1_avg, prauc_avg = self.get_performance_on_testset(encoder,
                                                                                                        decoder,
                                                                                                        patient_records_test)
            encoder.train()
            decoder.train()

            print(
                'epoch: {}; time: {}; Iteration: {}; jaccard_test: {:.4f}; precision_test: {:.4f}; recall_test: {:.4f}; f1_test: {:.4f}; prauc_test: {:.4f}'.format(
                    epoch, datetime.datetime.now(), trained_n_iteration, jaccard_avg, precision_avg, recall_avg, f1_avg,
                    prauc_avg))
            log_file.write(
                'epoch: {}; time: {}; Iteration: {}; jaccard_test: {:.4f}; precision_test: {:.4f}; recall_test: {:.4f}; f1_test: {:.4f}; prauc_test: {:.4f}\n'.format(
                    epoch, datetime.datetime.now(), trained_n_iteration, jaccard_avg, precision_avg, recall_avg, f1_avg,
                    prauc_avg))

            encoder_lr_scheduler.step(f1_avg)
            decoder_lr_scheduler.step(f1_avg)

            if epoch % save_every_epoch == 0:
                torch.save(
                    {'medrec_epoch': epoch,
                     'medrec_iteration': trained_n_iteration,
                     'encoder': encoder.state_dict(),
                     'decoder': decoder.state_dict(),
                     'encoder_optimizer': encoder_optimizer.state_dict(),
                     'decoder_optimizer': decoder_optimizer.state_dict()},
                    os.path.join(save_model_path,
                                 'medrec_{}_{}_{:.4f}.checkpoint'.format(epoch, trained_n_iteration, f1_avg)))

        log_file.close()

    def train(self, input_size, hidden_size, encoder_n_layers, encoder_embedding_dropout_rate,
              encoder_gru_dropout_rate, encoder_learning_rate, decoder_dropout_rate, decoder_hop_count,
              regular_hop_count, attn_type_kv, attn_type_embedding, least_adm_count, coverage_dim,
              decoder_learning_rate, loss_proportion_bce, loss_proportion_multi, save_model_dir='data/model',
              train_ratio=params.TRAIN_RATIO, test_ratio=params.TEST_RATIO, n_epoch=50, print_every_iteration=100,
              save_every_epoch=1, load_model_name=None):
        print('initializing >>>')

        if load_model_name:
            print('load model from checkpoint file: ', load_model_name)
            checkpoint = torch.load(load_model_name)

        encoder = EncoderLinearQuery(self.device, input_size, hidden_size, self.diagnose_count,
                                     self.procedure_count, encoder_n_layers, encoder_embedding_dropout_rate,
                                     encoder_gru_dropout_rate)

        decoder = DecoderGRUCoverFixed(self.device, hidden_size, self.medication_count, decoder_dropout_rate,
                                       least_adm_count, decoder_hop_count, coverage_dim, attn_type_kv,
                                       attn_type_embedding, regular_hop_count, self.ehr_matrix)

        if load_model_name:
            encoder_sd = checkpoint['encoder']
            decoder_sd = checkpoint['decoder']
            encoder.load_state_dict(encoder_sd)
            decoder.load_state_dict(decoder_sd)
        encoder = encoder.to(self.device)
        decoder = decoder.to(self.device)
        encoder.train()
        decoder.train()

        print('build optimizer >>>')
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=encoder_learning_rate)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=decoder_learning_rate)
        if load_model_name:
            encoder_optimizer_sd = checkpoint['encoder_optimizer']
            decoder_optimizer_sd = checkpoint['decoder_optimizer']
            encoder_optimizer.load_state_dict(encoder_optimizer_sd)
            decoder_optimizer.load_state_dict(decoder_optimizer_sd)

        print('start training >>>')
        patient_records = pd.read_pickle(self.patient_records_file)
        split_point = int(len(patient_records) * train_ratio)
        test_count = int(len(patient_records) * test_ratio)
        patient_records_train = patient_records[:split_point]
        patient_records_test = patient_records[split_point:split_point + test_count]

        medrec_trained_epoch = 0
        medrec_trained_iteration = 0

        if load_model_name:
            medrec_trained_n_epoch_sd = checkpoint['medrec_epoch']
            medrec_trained_n_iteration_sd = checkpoint['medrec_iteration']
            medrec_trained_epoch = medrec_trained_n_epoch_sd
            medrec_trained_iteration = medrec_trained_n_iteration_sd

        save_model_structure = str(encoder_n_layers) + '_' + str(input_size) + '_' + str(hidden_size)
        save_model_parameters = str(encoder_embedding_dropout_rate) + '_' + str(encoder_gru_dropout_rate) + '_' + str(
            decoder_dropout_rate) + '_' + attn_type_kv + '_' + attn_type_embedding + '_' + str(
            decoder_hop_count) + '_' + str(regular_hop_count)
        save_model_path = os.path.join(save_model_dir, save_model_structure, save_model_parameters)

        self.trainIters(encoder, decoder, encoder_optimizer, decoder_optimizer, patient_records_train,
                        patient_records_test, save_model_path, n_epoch, loss_proportion_bce, loss_proportion_multi,
                        print_every_iteration, save_every_epoch, medrec_trained_epoch, medrec_trained_iteration)


class AdaphopLastQueryTraining:
    def __init__(self, device, patient_records_file, voc_file, ehr_matrix_file):
        self.device = device
        self.patient_records_file = patient_records_file
        self.voc_file = voc_file
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

    def loss_function(self, hop_output, target_hop):
        target = np.zeros((1, GRU_COVER_MAX_HOP))
        target[:, target_hop] = 1
        loss = F.binary_cross_entropy_with_logits(hop_output, torch.FloatTensor(target).to(self.device))
        return loss

    def get_performance_on_testest(self, encoder, decoder, patient_records):
        jaccard_avg, precision_avg, recall_avg, f1_avg, prauc_avg = [], [], [], [], []
        for patient in patient_records:
            for idx, adm in enumerate(patient):
                current_records = patient[:idx + 1]
                target_medications = adm[params.MEDICATION_INDEX]
                target_multi_hot = np.zeros(self.medication_count)
                target_multi_hot[target_medications] = 1

                query, memory_keys, memory_values, hop_tag_gru_cover = encoder(current_records)
                predict_output, _ = decoder(query, memory_keys, memory_values)
                target_medications = adm[params.MEDICATION_INDEX]
                target_multi_hot = np.zeros(self.medication_count)
                target_multi_hot[target_medications] = 1
                predict_prob = torch.sigmoid(predict_output).detach().cpu().numpy()[0]
                predict_multi_hot = predict_prob.copy()

                index_nan = np.argwhere(np.isnan(predict_multi_hot))
                if index_nan.shape[0] != 0:
                    predict_multi_hot = np.zeros_like(predict_multi_hot)

                predict_multi_hot[predict_multi_hot >= 0.5] = 1
                predict_multi_hot[predict_multi_hot < 0.5] = 0
                predict_medications = list(np.where(predict_multi_hot == 1)[0])

                jaccard = self.evaluate_utils.metric_jaccard_similarity(predict_medications, target_medications)
                precision = self.evaluate_utils.metric_precision(predict_medications, target_medications)
                recall = self.evaluate_utils.metric_recall(predict_medications, target_medications)
                f1 = self.evaluate_utils.metric_f1(precision, recall)
                prauc = self.evaluate_utils.precision_auc(predict_prob, target_multi_hot)

                jaccard_avg.append(jaccard)
                precision_avg.append(precision)
                recall_avg.append(recall)
                f1_avg.append(f1)
                prauc_avg.append(prauc)

        jaccard_avg = np.mean(np.array(jaccard_avg))
        precision_avg = np.mean(np.array(precision_avg))
        recall_avg = np.mean(np.array(recall_avg))
        f1_avg = np.mean(np.array(f1_avg))
        prauc_avg = np.mean(np.array(prauc_avg))

        return jaccard_avg, precision_avg, recall_avg, f1_avg, prauc_avg

    def trainIters(self, encoder, decoder, decoder_optimizer, patient_records_train, patient_records_test,
                   save_model_path, n_epoch, print_every_iteration=100, save_every_epoch=5, trained_epoch=0,
                   trained_iteration=0):
        start_epoch = trained_epoch + 1
        trained_n_iteration = trained_iteration
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)
        log_file = open(os.path.join(save_model_path, 'adap_medrec_loss.log'), 'a+')
        decoder_lr_scheduler = ReduceLROnPlateau(decoder_optimizer, mode='max', patience=5, factor=0.1)

        for epoch in range(start_epoch, start_epoch + n_epoch):
            print_loss = []
            iteration = 0
            for patient in patient_records_train:
                current_records = patient
                decoder_optimizer.zero_grad()
                query, memory_keys, memory_values, hop_tag_gru_cover = encoder(current_records)
                if memory_keys is not None:
                    trained_n_iteration += 1
                    iteration += 1
                    _, hop_output = decoder(query, memory_keys, memory_values)
                    loss = self.loss_function(hop_output, hop_tag_gru_cover)
                    print_loss.append(loss.item())
                    loss.backward()
                    decoder_optimizer.step()

                    if iteration % print_every_iteration == 0:
                        print_loss_avg = np.mean(np.array(print_loss))
                        print_loss = []
                        print(
                            'epoch: {}; time: {}; Iteration: {}; train loss: {:.4f}'.format(
                                epoch, datetime.datetime.now(), trained_n_iteration, print_loss_avg))
                        log_file.write(
                            'epoch: {}; time: {}; Iteration: {};  train loss: {:.4f}\n'.format(
                                epoch, datetime.datetime.now(), trained_n_iteration, print_loss_avg))

            decoder.eval()
            jaccard_avg, precision_avg, recall_avg, f1_avg, prauc_avg = self.get_performance_on_testest(encoder,
                                                                                                        decoder,
                                                                                                        patient_records_test)
            decoder.train()
            print(
                'epoch: {}; time: {}; Iteration: {}; jaccard_test: {:.4f}; precision_test: {:.4f}; recall_test: {:.4f}; f1_test: {:.4f}; prauc_test: {:.4f}'.format(
                    epoch, datetime.datetime.now(), trained_n_iteration, jaccard_avg, precision_avg, recall_avg, f1_avg,
                    prauc_avg))
            log_file.write(
                'epoch: {}; time: {}; Iteration: {}; jaccard_test: {:.4f}; precision_test: {:.4f}; recall_test: {:.4f}; f1_test: {:.4f}; prauc_test: {:.4f}\n'.format(
                    epoch, datetime.datetime.now(), trained_n_iteration, jaccard_avg, precision_avg, recall_avg, f1_avg,
                    prauc_avg))
            decoder_lr_scheduler.step(f1_avg)
            if epoch % save_every_epoch == 0:
                torch.save({'adap_medrec_epoch': epoch,
                            'adap_medrec_iteration': trained_n_iteration,
                            'encoder': encoder.state_dict(),
                            'decoder': decoder.state_dict(),
                            'decoder_optimizer': decoder_optimizer.state_dict()},
                           os.path.join(save_model_path,
                                        'adap_medrec_{}_{}_{:.4f}.checkpoint'.format(epoch, trained_n_iteration,
                                                                                     f1_avg)))
        log_file.close()

    def train(self, input_size, hidden_size, encoder_n_layers, encoder_embedding_dropout_rate, encoder_gru_dropout_rate,
              decoder_dropout_rate, decoder_max_hop_count, regular_hop_count, attn_type_kv, attn_type_embedding,
              least_adm_count, coverage_dim, decoder_learning_rate, load_model_file, load_model_type='fixed',
              save_model_dir='data/model', n_epoch=50, print_every_iteration=100, save_every_epoch=1):
        print('initializing >>>')

        print('load fixed hop model from:', load_model_file)
        checkpoint = torch.load(load_model_file)

        encoder = EncoderLinearAdap(self.device, input_size, hidden_size, self.diagnose_count, self.procedure_count,
                                    encoder_n_layers, encoder_embedding_dropout_rate, encoder_gru_dropout_rate)
        decoder = DecoderGRUCoverAdapLastQuery(self.device, hidden_size, self.medication_count,
                                               decoder_dropout_rate, least_adm_count, decoder_max_hop_count,
                                               coverage_dim, attn_type_kv, attn_type_embedding, regular_hop_count,
                                               self.ehr_matrix)

        encoder_sd = checkpoint['encoder']
        decoder_sd = checkpoint['decoder']
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd, strict=False)
        for net_param in encoder.parameters():
            net_param.requires_grad = False
        for net_param in decoder.parameters():
            net_param.requires_grad = False
        decoder.hop_count_output = nn.Sequential(nn.ReLU(), nn.Linear(hidden_size, decoder_max_hop_count),
                                                 nn.LogSoftmax(dim=1))
        encoder = encoder.to(self.device)
        decoder = decoder.to(self.device)
        encoder.eval()
        decoder.train()

        print('build optimizer >>>')
        decoder_optimizer = optim.Adam(decoder.hop_count_output.parameters(), lr=decoder_learning_rate)
        if load_model_type == 'adap':
            decoder_optimizer_sd = checkpoint['decoder_optimizer']
            decoder_optimizer.load_state_dict(decoder_optimizer_sd)

        print('load data >>>')
        patient_records = pd.read_pickle(self.patient_records_file)
        patient_records_train = patient_records['training']
        patient_records_test = patient_records['test']

        print('start training >>>')
        adap_medrec_trained_epoch = 0
        adap_medrec_trained_iteration = 0
        if load_model_type == 'adap':
            adap_medrec_trained_n_epoch_sd = checkpoint['adap_medrec_epoch']
            adap_medrec_trained_n_iteration_sd = checkpoint['adap_medrec_iteration']
            adap_medrec_trained_epoch = adap_medrec_trained_n_epoch_sd
            adap_medrec_trained_iteration = adap_medrec_trained_n_iteration_sd

        save_model_structure = str(encoder_n_layers) + '_' + str(input_size) + '_' + str(hidden_size)
        save_model_parameters = str(encoder_embedding_dropout_rate) + '_' + str(encoder_gru_dropout_rate) + '_' + str(
            decoder_dropout_rate) + '_' + attn_type_kv + '_' + attn_type_embedding + '_' + str(
            decoder_max_hop_count) + '_' + str(regular_hop_count)
        save_model_path = os.path.join(save_model_dir, save_model_structure, save_model_parameters)

        self.trainIters(encoder, decoder, decoder_optimizer, patient_records_train, patient_records_test,
                        save_model_path, n_epoch, print_every_iteration, save_every_epoch, adap_medrec_trained_epoch,
                        adap_medrec_trained_iteration)


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

    def evaluateIters(self, encoder, decoder, patient_records, save_result_path=None):
        if not os.path.exists(save_result_path):
            os.makedirs(save_result_path)
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
                  file=open(os.path.join(save_result_path, 'predict_result.pkl'), 'wb'))

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

        print('start evaluation >>>')
        self.evaluateIters(encoder, decoder, patient_records_validation, save_result_path)


class GetSupervisedHopData:
    def __init__(self, device, patient_records_file, concept2id_file, ehr_matrix_file):
        self.device = device
        self.patient_records_file = patient_records_file
        self.concept2id_file = concept2id_file
        self.ehr_matrix_file = ehr_matrix_file

        voc = dill.load(open(self.concept2id_file, 'rb'))
        self.diag_voc = voc['diag_voc']
        self.pro_voc = voc['pro_voc']
        self.med_voc = voc['med_voc']

        self.diagnose_count = len(self.diag_voc.word2idx)
        self.procedure_count = len(self.pro_voc.word2idx)
        self.medication_count = len(self.med_voc.word2idx)

        self.ehr_matrix = dill.load(open(self.ehr_matrix_file, 'rb'))

    def get_hop_data_dict(self, encoder, decoder, records):
        new_records = {}
        for i, patient in enumerate(tqdm(records)):
            for idx, adm in enumerate(patient):
                current_records = patient[:idx + 1]
                query, memory_keys, memory_values = encoder(current_records)
                target_medications = current_records[-1][params.MEDICATION_INDEX]
                hop_ground_truth, _, _ = decoder(query, memory_keys, memory_values, target_medications)
                current_records[-1].append(hop_ground_truth)
                if hop_ground_truth in new_records.keys():
                    new_records[hop_ground_truth].append(current_records)
                else:
                    new_records[hop_ground_truth] = [current_records]

        return new_records

    def get_hop_data(self, encoder, decoder, records):
        new_patient_records = []
        for i, patient in enumerate(tqdm(records)):
            current_patient = []
            for idx, adm in enumerate(patient):
                current_records = patient[:idx + 1]
                query, memory_keys, memory_values = encoder(current_records)
                target_medications = current_records[-1][params.MEDICATION_INDEX]
                hop_ground_truth, _, _ = decoder(query, memory_keys, memory_values, target_medications)
                adm.append(hop_ground_truth)
                current_patient.append(adm)
            new_patient_records.append(current_patient)
        return new_patient_records

    def resample_data(self, records_dict, count_dict):
        new_records = []
        records_dict.pop(-3)
        for (key, value) in records_dict.items():
            current_sample_count = len(value)
            target_sample_count = count_dict[key]
            current_records = []
            if current_sample_count > target_sample_count:
                for _ in range(target_sample_count):
                    current_records.append(random.sample(value, 1)[0])
            else:
                for _ in range(target_sample_count - current_sample_count):
                    current_records.append(random.sample(value, 1)[0])
                current_records = current_records + value

            new_records = new_records + current_records

        random.shuffle(new_records)
        return new_records

    def processIters(self, encoder, decoder, patient_records, records_train_ratio, records_test_ratio,
                     save_new_records_path, sample_count_dict):
        if not os.path.exists(save_new_records_path):
            os.makedirs(save_new_records_path)

        split_point = int(len(patient_records) * records_train_ratio)
        test_count = int(len(patient_records) * records_test_ratio)
        patient_records_train = patient_records[:split_point]
        patient_records_test = patient_records[split_point:split_point + test_count]
        patient_records_validation = patient_records[split_point + test_count:]

        patient_records_train_dict = self.get_hop_data_dict(encoder, decoder, patient_records_train)
        patient_records_test = self.get_hop_data(encoder, decoder, patient_records_test)
        patient_records_validation = self.get_hop_data(encoder, decoder, patient_records_validation)

        resample_records_training = self.resample_data(patient_records_train_dict, sample_count_dict)
        dill.dump(obj={'training': resample_records_training, 'test': patient_records_test,
                       'validation': patient_records_validation},
                  file=open(os.path.join(save_new_records_path, 'records_supervised_hop.pkl'), 'wb'))

    def process(self, load_model_name, input_size, hidden_size, encoder_n_layers, encoder_embedding_dropout_rate,
                encoder_gru_dropout_rate, decoder_dropout_rate, decoder_hop_count, regular_hop_count, attn_type_kv,
                attn_type_embedding, least_adm_count, coverage_dim, records_train_ratio, records_test_ratio,
                sample_count_dict, save_new_records_path):
        print('load model from checkpoint file:', load_model_name)
        checkpoint = torch.load(load_model_name)

        encoder = EncoderLinearQuery(self.device, input_size, hidden_size, self.diagnose_count, self.procedure_count,
                                     encoder_n_layers, encoder_embedding_dropout_rate, encoder_gru_dropout_rate)
        decoder = DecoderGRUCoverFixedCheckTruth(params.device, hidden_size, self.medication_count,
                                                 decoder_dropout_rate, least_adm_count, decoder_hop_count,
                                                 coverage_dim, attn_type_kv, attn_type_embedding, regular_hop_count,
                                                 ehr_adj=self.ehr_matrix)

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

        print('start >>>')
        self.processIters(encoder, decoder, patient_records, records_train_ratio, records_test_ratio,
                          save_new_records_path, sample_count_dict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patient_records_file', default=params.PATIENT_RECORDS_FILE, type=str, required=False)
    parser.add_argument('--patient_records_file_supervised_hop', default=params.PATIENT_RECORDS_FILE_SUPERVISED_HOP,
                        type=str, required=False)
    parser.add_argument('--voc_file', default=params.CONCEPTID_FILE, type=str, required=False)
    parser.add_argument('--ehr_matrix_file', default=params.EHR_MATRIX_FILE, type=str, required=False)

    parser.add_argument('--mode', default='fixed_hop_train',
                        type=str)  # fixed_hop_train,generate_adaptive_hop,adaptive_hop_train,eval

    # parameters of the encoder
    parser.add_argument('--hidden_size', default=200, type=int)
    parser.add_argument('--gru_n_layers', default=1, type=int)
    parser.add_argument('--gru_dropout_rate', default=0.04281694, type=float)
    parser.add_argument('--embedding_dropout_rate', default=0.31037152, type=float)
    parser.add_argument('--encoder_lr', default=0.000037, type=float)

    # parameters of the decoder
    parser.add_argument('--decoder_dropout_rate', default=0.77016361, type=float)
    parser.add_argument('--least_adm_count', default=13, type=int)
    parser.add_argument('--max_hop_count', default=14, type=int)
    parser.add_argument('--regular_hop_count', default=6, type=int)
    parser.add_argument('--coverage_dim', default=17, type=int)
    parser.add_argument('--attn_type_kv', default='dot', type=str)
    parser.add_argument('--attn_type_embedding', default='general', type=str)
    parser.add_argument('--decoder_lr', default=0.000095, type=float)

    parser.add_argument('--loss_proportion_bce', default=params.LOSS_PROPORTION_BCE, type=float)
    parser.add_argument('--loss_proportion_multi', default=params.LOSS_PROPORTION_Multi_Margin, type=float)
    parser.add_argument('--records_train_ratio', default=params.TRAIN_RATIO, type=float)
    parser.add_argument('--records_test_ratio', default=params.TEST_RATIO, type=float)
    parser.add_argument('--predict_prob_threshold', default=0.5, type=float)

    parser.add_argument('--save_model_dir', default='data/model/FixedhopModel', type=str, required=False)
    parser.add_argument('--fixed_hop_n_epoch', default=40, type=int)
    parser.add_argument('--adaptive_hop_n_epoch', default=10, type=int)
    parser.add_argument('--print_every_iteration', default=100, type=int)
    parser.add_argument('--save_every_epoch', default=1, type=int)
    parser.add_argument('--load_model_name', default=None, type=str)
    parser.add_argument('--load_model_type', default='fixed', type=str)

    parser.add_argument('--save_supervised_hop_dir', default='data', type=str)
    parser.add_argument('--save_predict_results_dir', default='data/predict_results', type=str)

    args = parser.parse_args()

    sample_count_dict = {0: 700, 1: 700, 2: 600, 3: 600, 4: 600, 5: 600, 6: 600, 7: 600, 8: 600, 9: 600, 10: 600,
                         11: 600, 12: 600, 13: 600}

    if args.mode not in ['fixed_hop_train', 'generate_adaptive_hop', 'adaptive_hop_train', 'eval']:
        print('choose mode from fixed_hop_train,generate_adaptive_hop,adaptive_hop_train and eval')
        return

    if args.mode in ['generate_adaptive_hop', 'adaptive_hop_train', 'eval'] and args.load_model_name is None:
        print(
            '--load_model_name is required if you choose generate_adaptive_hop,adaptive_hop_train or eval as the --mode')
        return

    if args.mode == 'fixed_hop_train':
        module = FixedHopTraining(params.device, args.patient_records_file, args.voc_file, args.ehr_matrix_file)
        module.train(args.hidden_size, args.hidden_size, args.gru_n_layers, args.embedding_dropout_rate,
                     args.gru_dropout_rate, args.encoder_lr, args.decoder_dropout_rate, args.max_hop_count,
                     args.regular_hop_count, args.attn_type_kv, args.attn_type_embedding, args.least_adm_count,
                     args.coverage_dim, args.decoder_lr, args.loss_proportion_bce, args.loss_proportion_multi,
                     args.save_model_dir, args.records_train_ratio, args.records_test_ratio, args.fixed_hop_n_epoch,
                     args.print_every_iteration, args.save_every_epoch, args.load_model_name)

    if args.mode == 'generate_adaptive_hop':
        module = GetSupervisedHopData(params.device, args.patient_records_file, args.voc_file, args.ehr_matrix_file)
        module.process(args.load_model_name, args.hidden_size, args.hidden_size, args.gru_n_layers,
                       args.embedding_dropout_rate, args.gru_dropout_rate, args.decoder_dropout_rate,
                       args.max_hop_count, args.regular_hop_count, args.attn_type_kv, args.attn_type_embedding,
                       args.least_adm_count, args.coverage_dim, args.records_train_ratio, args.records_test_ratio,
                       sample_count_dict, args.save_supervised_hop_dir)

    if args.mode == 'adaptive_hop_train':
        module = AdaphopLastQueryTraining(params.device, args.patient_records_file_supervised_hop, args.voc_file,
                                          args.ehr_matrix_file)
        module.train(args.hidden_size, args.hidden_size, args.gru_n_layers, args.embedding_dropout_rate,
                     args.gru_dropout_rate, args.decoder_dropout_rate, args.max_hop_count, args.regular_hop_count,
                     args.attn_type_kv, args.attn_type_embedding, args.least_adm_count, args.coverage_dim,
                     args.decoder_lr, args.load_model_name, args.load_model_type, args.save_model_dir,
                     args.adaptive_hop_n_epoch, args.print_every_iteration, args.save_every_epoch)

    if args.mode == 'eval':
        module = AdaphopLastQueryEvaluation(params.device, args.voc_file, args.patient_records_file_supervised_hop,
                                            args.predict_prob_threshold, args.ehr_matrix_file)
        module.evaluate(args.load_model_name, args.hidden_size, args.hidden_size, args.gru_n_layers,
                        args.embedding_dropout_rate, args.gru_dropout_rate, args.decoder_dropout_rate,
                        args.max_hop_count, args.regular_hop_count, args.attn_type_kv, args.attn_type_embedding,
                        args.least_adm_count, args.coverage_dim, args.save_predict_results_dir)


if __name__ == '__main__':
    main()
