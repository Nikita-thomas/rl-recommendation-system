#Dependencies
import pandas as pd
import numpy as np
import tensorflow as tf
import itertools
import matplotlib.pyplot as plt
import time

from envs import OfflineEnv
from recommender import DRRAgent

import os
ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'ml-1m/ml-1m')
STATE_SIZE = 10
MAX_EPISODE_NUM = 10

# os.environ["CUDA_VISIBLE_DEVICES"]="1"

if __name__ == "__main__":

    print('Data loading...')

    #Loading datasets
    #ratings_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR,'ratings.dat'), 'r').readlines()]
    #users_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR,'users.dat'), 'r').readlines()]
    #movies_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR,'movies.dat'),encoding='latin-1').readlines()]
    
    users_df = pd.read_csv('./data/BX-Users.csv', sep=';', encoding='latin-1')
    ratings_df = pd.read_csv('./data/BX-Book-Ratings.csv', sep=';', encoding='latin-1')
    books_df = pd.read_csv('./data/Books.csv', sep=';', encoding='latin-1')
    books_df = books_df[['ISBN','Title', 'Author']]

    print("Data loading complete!")
    print("Data preprocessing...")
    
    books_id_to_books = {book[0]: book[1] for book in books_df[['ISBN', 'Title']].values}
    ratings_df['ISBN'] = ratings_df['ISBN'].map(books_id_to_books)

    # 유저별로 본 영화들 순서대로 정리
    users_dict = np.load('./data/user_dict_books.npy', allow_pickle=True)

    # 각 유저별 영화 히스토리 길이
    users_history_lens = np.load('./data/usesr_history_len_books.npy')

    users_num = max(ratings_df["User-ID"])+1
    items_num = (ratings_df["ISBN"].nunique())+1

    unique_users = ratings_df['User-ID'].unique()[:6040]
    ratings_df = ratings_df[ratings_df['User-ID'].isin(unique_users)]

    # Training setting
    train_users_num = int(users_num * 0.8)
    train_items_num = items_num
    train_users_dict = {k:users_dict.item().get(k) for k in range(1, train_users_num+1)}
    train_users_history_lens = users_history_lens[:train_users_num]
    
    print('DONE!')
    time.sleep(2)

    env = OfflineEnv(train_users_dict, train_users_history_lens, books_id_to_books, STATE_SIZE)
    recommender = DRRAgent(env, users_num, items_num, STATE_SIZE, use_wandb=False)
    recommender.actor.build_networks()
    recommender.critic.build_networks()
    recommender.train(MAX_EPISODE_NUM, load_model=False)