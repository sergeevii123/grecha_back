import pandas as pd
import numpy as np
import re
import gensim
from sklearn import cluster
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from scipy.sparse import csr_matrix
from implicit.nearest_neighbours import CosineRecommender
import pickle
import scipy


def preprocess_text(text):
    text = re.sub('[^a-zA-Zа-яА-Я]+', ' ', text)
    text = re.sub(' +', ' ', text)
    return text.strip().split(' ')


def prepare_for_w2v(list_of_names):
    out = []
    for sentence in list_of_names:
        out.append(preprocess_text(sentence.lower()))
    return out


def name_to_cluster_id(name):
    return labels[unique_names_list.index(name)]


def cluster_id_to_cluster_name(id_):
    return cluster_to_cluster_name[id_][0]


def sent_vectorizer(sent, model):
    sent_vec = []
    numw = 0
    for w in sent:
        try:
            if numw == 0:
                sent_vec = model[w]
            else:
                sent_vec = np.add(sent_vec, model[w])
            numw += 1
        except:
            print('here')

    return np.asarray(sent_vec) / numw


if __name__ == '__main__':
    print('read data')
    services_df = pd.read_csv('data/services_hackaton.csv', sep=';')
    pupil_df = pd.read_csv('data/Pupil_hackaton.csv', sep=';')
    mega_relation_df = pd.read_csv('data/MegaRelation_hackaton.csv', sep=';')

    print('cluster services')
    services_df = services_df[services_df['Наименование_услуги'].notna()].reset_index(drop=True)

    prepared_sentences = prepare_for_w2v(services_df['Наименование_услуги'].unique())

    w2v = gensim.models.Word2Vec(prepared_sentences, min_count=1, size=32)

    X = []
    for sentence in prepared_sentences:
        X.append(sent_vectorizer(sentence, w2v))

    kmeans = cluster.KMeans(n_clusters=1000, n_jobs=40)
    kmeans.fit(X)

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    unique_names_list = list(services_df['Наименование_услуги'].unique())

    services_df['id_clustered'] = services_df['Наименование_услуги'].apply(name_to_cluster_id)

    cluster_to_cluster_name = dict()
    for i in range(1000):
        cur_name = \
        services_df[services_df['id_clustered'] == i]['Наименование_услуги'].value_counts().nlargest(1).index[0]
        cluster_to_cluster_name[i] = (cur_name, len(services_df[services_df['id_clustered'] == i]))

    services_df['clustered_name'] = services_df['id_clustered'].apply(cluster_id_to_cluster_name)

    print('working with mega_relation_df')
    mega_relation_df['Дата_создания_записи'] = pd.to_datetime(mega_relation_df['Дата_создания_записи'])
    mega_relation_df = mega_relation_df[mega_relation_df['id_ученика'].notna()].reset_index(drop=True)
    mega_relation_df['id_ученика'] = mega_relation_df['id_ученика'].astype(int)
    mega_relation_df = mega_relation_df[mega_relation_df['id_заявления'].notna()].reset_index(drop=True)
    mega_relation_df['id_заявления'] = mega_relation_df['id_заявления'].astype(int)
    pupil_df = pupil_df.drop_duplicates('id_ученика').reset_index(drop=True)
    pupil_df = pupil_df[pupil_df['возраст'] > 0].reset_index(drop=True)
    pupil_df['возраст'] = pupil_df['возраст'].astype(int)
    pupil_df = pupil_df[pupil_df['пол'].notna()].reset_index(drop=True)

    pupil_gender_le = LabelEncoder()

    pupil_gender_le.fit(pupil_df['пол'])
    pupil_df['пол'] = pupil_gender_le.transform(pupil_df['пол'])

    pupil_to_many_kdf = pupil_df.set_index('id_ученика').join(mega_relation_df.set_index('id_ученика')).reset_index()
    pupil_to_many_kdf = pupil_to_many_kdf[pupil_to_many_kdf['id_услуги'].notna()].reset_index()

    pupil_to_many_kdf['id_услуги'] = pupil_to_many_kdf['id_услуги'].astype(int)

    services_df = services_df[services_df['Наименование_услуги'].notna()].reset_index(drop=True)

    cols_with_service_info = ['Тип_финансирования', 'Наименование_услуги']
    le_services_df = dict()
    for col in cols_with_service_info:
        print(col)
        le = LabelEncoder()
        le.fit(services_df[col])
        services_df[col] = le.transform(services_df[col])
        le_services_df[col] = le

    pupil_to_many_kdf_with_service_info = pupil_to_many_kdf.set_index(['id_организации', 'id_услуги']). \
        join(services_df.set_index(['id_организации', 'id_услуги'])).reset_index()

    print('filter time')
    pupil_to_many_kdf_with_service_info = \
        pupil_to_many_kdf_with_service_info[
            pupil_to_many_kdf_with_service_info['Дата_создания_записи'] >= '2016-01-01'].reset_index(drop=True)

    pupil_to_many_kdf_with_service_info = \
        pupil_to_many_kdf_with_service_info[
            pupil_to_many_kdf_with_service_info['Дата_создания_записи'] < '2020-01-01'].reset_index(drop=True)

    pupil_to_many_kdf_with_service_info['id_организации'] = pupil_to_many_kdf_with_service_info[
        'id_организации'].astype(int)
    pupil_to_many_kdf_with_service_info = pupil_to_many_kdf_with_service_info[
        pupil_to_many_kdf_with_service_info['id_clustered'].notna()].reset_index(drop=True)

    pupil_to_many_kdf_with_service_info['id_clustered'] = pupil_to_many_kdf_with_service_info['id_clustered'].astype(
        int)

    pupil_to_many_kdf_with_service_info = pupil_to_many_kdf_with_service_info[
        pupil_to_many_kdf_with_service_info.clustered_name != ' '].reset_index(drop=True)
    pupil_to_many_kdf_with_service_info = pupil_to_many_kdf_with_service_info[
        pupil_to_many_kdf_with_service_info.clustered_name != '4 часа в неделю'].reset_index(drop=True)

    pupil_to_many_kdf_with_service_info = pupil_to_many_kdf_with_service_info[
        pupil_to_many_kdf_with_service_info.clustered_name != 'уровень I'].reset_index(drop=True)

    pupil_to_many_kdf_with_service_info = pupil_to_many_kdf_with_service_info[
        pupil_to_many_kdf_with_service_info.clustered_name != 'уровень II'].reset_index(drop=True)

    pupil_to_many_kdf_with_service_info = pupil_to_many_kdf_with_service_info[
        pupil_to_many_kdf_with_service_info.clustered_name != '6 часов в неделю'].reset_index(drop=True)

    pupil_to_many_kdf_with_service_info = pupil_to_many_kdf_with_service_info[
        pupil_to_many_kdf_with_service_info.clustered_name != '(обновлённый универсальный шаблон)'].reset_index(
        drop=True)

    pupil_to_many_kdf_with_service_info = pupil_to_many_kdf_with_service_info[
        pupil_to_many_kdf_with_service_info.clustered_name != \
        'Шаблон общеразвивающей программы (4 часа занятий в неделю)'].reset_index(
        drop=True)

    most_popular_items_names = Counter(pupil_to_many_kdf_with_service_info['clustered_name'])

    with open('popular_clusters', 'w') as f:
        for key, value in most_popular_items_names.most_common(500):
            f.write(f'{key}:{value}\n')

    print('train recommendation')
    train_df = pupil_to_many_kdf_with_service_info[
        pupil_to_many_kdf_with_service_info['Дата_создания_записи'] < '2019-12-01']
    test_df = pupil_to_many_kdf_with_service_info[
        pupil_to_many_kdf_with_service_info['Дата_создания_записи'] >= '2020-12-01']

    user_column = 'id_ученика'
    item_column = 'id_clustered'

    train_pairs = train_df[[user_column, item_column]]
    test_pairs = test_df[[user_column, item_column]]

    test_pairs = test_pairs[test_pairs[user_column].isin(set(train_pairs[user_column].unique()))].reset_index(drop=True)
    test_pairs = test_pairs[test_pairs[item_column].isin(set(train_pairs[item_column].unique()))].reset_index(drop=True)

    leusers = LabelEncoder()
    train_pairs[user_column] = leusers.fit_transform(train_pairs[user_column])
    leservices = LabelEncoder()
    train_pairs[item_column] = leservices.fit_transform(train_pairs[item_column])

    test_pairs[user_column] = leusers.transform(test_pairs[user_column])
    test_pairs[item_column] = leservices.transform(test_pairs[item_column])

    n_users = len(leusers.classes_)
    n_items = len(leservices.classes_)

    sparse_matrix = csr_matrix(
        (np.ones(len(train_pairs)), (train_pairs[user_column], train_pairs[item_column])),
        shape=(n_users, n_items)
    )

    model = CosineRecommender()
    model.fit(sparse_matrix.T)

    print('saving artifacts')
    with open('leservices.pkl', 'wb') as f:
        pickle.dump(leservices, f)
    with open('kdf_rec.pkl', 'wb') as f:
        pickle.dump(model, f)
    scipy.sparse.save_npz('sparse_kdf.npz', sparse_matrix)
    services_df_for_save = services_df[
        services_df.id_clustered.isin(leservices.inverse_transform(train_pairs[item_column]))].reset_index(drop=True)
    services_df_for_save = services_df_for_save.drop_duplicates('id_clustered').reset_index(drop=True)
    services_df_for_save['id_enc_cluster'] = leservices.transform(services_df_for_save.id_clustered)
    services_df.to_csv('services.csv', index=False)

    print('finished')
