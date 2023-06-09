# coding: utf-8
# 必要なモジュールのインポート
import random
import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer, BertModel
from sentence_transformers import SentenceTransformer, util


from glob import glob
from natsort import natsorted

from flask import Flask, request, render_template
from wtforms import Form, StringField, validators, SubmitField, TextAreaField



# Flask をインスタンス化
app = Flask(__name__)

#　類似文章を抽出
def similartexts(input_text):

    # SBERTの英語モデル
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 入力テキストの文章ベクトルの格納先
    input_text_vectors = []

    # 入力テキストの文章ベクトルを計算
    embedding = model.encode(input_text, convert_to_tensor=False)
    input_text_vectors.append(embedding)
    # numpy.ndarrayにする
    input_text_vectors = np.vstack(input_text_vectors)

    # 類似度計算
    # sentence_vectors.csvファイル(突き合わせ対象の文章ベクトル)を読み込み
    #sentence_vectors=np.loadtxt('./src/sentence_vectors.csv', delimiter=',')
    sentence_vectors=np.loadtxt('./sentence_vectors.csv', delimiter=',')
    # job.csvを読み込み
    #df_job = pd.read_csv('./src/job.csv', index_col=0)
    df_job = pd.read_csv('./job.csv', index_col=0)
    df_job.drop(['Job Description'], axis=1, inplace=True)
    # 先にノルムを1にしておく。
    norm = np.linalg.norm(sentence_vectors, axis=1, keepdims=True) 
    sentence_vectors_normalized = sentence_vectors / norm
    input_norm = np.linalg.norm(input_text_vectors, axis=1, keepdims=True)
    input_text_vectors_normalized = input_text_vectors / input_norm
    # 類似度行列を計算する
    sim_matrix = sentence_vectors_normalized.dot(input_text_vectors_normalized.T)
    sim_matrix = np.round(sim_matrix, decimals=3)
    # 類似度をdf_jobに結合
    df_sim = pd.DataFrame(sim_matrix, columns=['Similarity'])
    result_sim = df_job.join(df_sim)
    #df_match_top5 = result_sim.sort_values(by='Similarity', ascending=False).head(5)
    df_result =  result_sim.sort_values(by='Similarity', ascending=False)
 
    #return df_match_top5
    return df_result


# WTForms を使い、index.html 側で表示させるフォームを構築
class InputForm(Form):
    InputFormTest = TextAreaField('Please briefly explain why you are qualified for the jobs you are applying for based on your experience, achievements, and acquired skills.',
                    [validators.required()])

    # HTML 側で表示する submit ボタンの表示
    submit = SubmitField('Start Matching!!')

# URL にアクセスがあった場合の挙動の設定
@app.route('/', methods = ['GET', 'POST'])
def input():
    # WTForms で構築したフォームをインスタンス化
    form = InputForm(request.form)

    # POST メソッドの条件の定義
    if request.method == 'POST':

        # 条件に当てはまる場合
        if form.validate() == False:
            return render_template('index.html', forms=form)
        # 条件に当てはまらない場合の処理の実行を定義
        else:
            input_text = request.form['InputFormTest']
            df_output = similartexts(input_text)

            header = df_output.columns
            record = df_output.values.tolist()

            return render_template('result.html', header=header, record=record)

    # GET メソッドの定義
    elif request.method == 'GET':
        return render_template('index.html', forms=form)

# アプリケーションの実行の定義
if __name__ == '__main__':
    app.run(debug=True)
