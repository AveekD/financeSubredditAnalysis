# TO RUN, RUN python preprocess.py -n=[name of subreddit]
# OPTIONAL ARGUEMENTS OF -b=[start date, utc] and -e=[end date, utc], defaulted to 12/1/18 and 12/1/19
# NEED A /data/ folder IN THE SAME DIR AS preprocess.py, AND NEED TO HAVE ALL NECESSARY pkl FILES PRECREATED 
# BEFORE RUNNING (WILL WARN IF NOT ALREADY CREATED)

# THIS SCRIPT DOES THE FOLLOWING
# 1. TAKES IN SUBREDDIT NAME, START DATE, END DATE
# 2. DOWNLOADS ALL POSTS AND COMMENTS MADE IN SUBREDDIT DURING TIME FRAME
# 3. USE KEYWORD EXTRACTION CLASSIFICATION, IDENTIFY RELVAENT POST/COMMENT THREADS
# 4. RETURNS DATAFRAME OF POSTS / COMMENTS (PICKLED)

# IMPORT STATEMENTS
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import coo_matrix
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import pickle
from scraper import get_posts, get_comments
import argparse
import calendar
import datetime
from os import path

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
 
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
 
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results

def get_keywords(doc, tfidf_transformer, cv, feature_names):
    #generate tf-idf for the given document
    tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))

    #sort the tf-idf vectors by descending order of scores
    sorted_items=sort_coo(tf_idf_vector.tocoo())
    #extract only the top n; n here is 10
    keywords=extract_topn_from_vector(feature_names,sorted_items,5)
    return keywords

def main(subreddit_name, start_date, end_date):

    post_exists = path.exists('./data/{}-posts.pkl'.format(subreddit_name))
    comments_exists = path.exists('./data/{}-comments.pkl'.format(subreddit_name))
    results_exists = path.exists('./data/{}-results.pkl'.format(subreddit_name))
    if not (post_exists and comments_exists and results_exists):
        print('not all files available')
        print('make sure you have /data/{}-posts.pkl, /data/{}-comments.pkl'.format(subreddit_name, subreddit_name))
        print('and /data/{}-results.pkl created already'.format(subreddit_name))
        return

    # DOWNLOAD POSTS AND COMMENTS
    posts_list = get_posts(subreddit_name, start_date, end_date)
    df_posts = pd.DataFrame(posts_list)
    df_posts['text'] = df_posts['title'] + ' ' + df_posts['selftext']
    df_posts['used_id'] = df_posts['id']
    print('loaded posts...')

    comments_list = get_comments(subreddit_name, start_date, end_date)
    df_comments = pd.DataFrame(comments_list)
    df_comments['text'] = df_comments['body']
    df_comments = df_comments[df_comments['author'] != 'AutoModerator']
    df_comments['used_id'] = df_comments['link_id'].str.slice(3)
    print('loaded comments...')

    # COMPILE POSTS AND COMMENTS INTO THREADS

    threads = {}
    for index, x in df_posts.iterrows():
        threads[str(x.used_id)] = str(x.title) + ' ' + str(x.selftext)
    for index, x in df_comments.iterrows():
        if str(x.used_id) in threads:
            threads[str(x.used_id)] += ' ' + str(x.body)
    print('combined into post-comment threads...')

    # CONVERT THREADS INTO A 'CORPUS' FOR KEYWORD MODEL

    stop_words = set(stopwords.words('english'))
    new_stop_words = set(['removed','x200b', 'amp'])
    stop_words = stop_words.union(new_stop_words)
    corpus = []
    corpus_w_key = {}
    tokenizer = RegexpTokenizer(r'\w+')
    for key in threads:
        x = threads[key]
        text = str(x.lower())
        text = tokenizer.tokenize(re.sub(r'https?://\S+', '', text))
        ps=PorterStemmer()
        lem = WordNetLemmatizer()
        text = [lem.lemmatize(word) for word in text if not word in  
                stop_words] 
        text = " ".join(text)
        corpus.append(text)
        corpus_w_key[key] = text
    print('cleaned post-comment threads...')

    # FITTING KEYWORD GENERATOR (ML STUFFZ)

    cv = CountVectorizer(max_df=0.8,stop_words=stop_words, max_features=10000, ngram_range=(1,3))
    X = cv.fit_transform(corpus)
    tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(X)
    feature_names=cv.get_feature_names()
    print('train keyword generator model...')

    # GET LIST OF RELAVENT IDS
    valid_ids = []
    target_keywords = ['market','stock', 'long term', 'calls', 'puts', 'invest', 'roth ira', 'buy', 'sell', 'growth', 'etf', 'dow', 'short term', 'stock market', 'short', 'bear', 'bull', 'correction', 'recession', 'total stock market', 'trade war', '10 year', 'stock trading','market index fund' , 'interest rate', 'index fund']
    for key in corpus_w_key:
        text = corpus_w_key[key]
        keywords = get_keywords(text, tfidf_transformer, cv, feature_names)
        for target in target_keywords:
            if target in keywords:
                valid_ids.append(key)
                break
    print('obtained list of relavent thread-ids...')

    # RETURN PICKLE FORMAT OF POSTS / COMMENTS THAT ARE RELAVENT
    df_comments['type'] = 'comment'
    df_posts['type'] = 'post'
    x = df_comments[['created_utc','used_id','text','score','type']]
    y = df_posts[['created_utc','used_id','text','score','type']]
    all_data = x.append(y,ignore_index=True)
    is_valid = []
    for x in all_data['used_id']:
        if x in valid_ids:
            is_valid.append(True)
        else:
            is_valid.append(False)
    all_data = all_data[is_valid]
    all_data.to_pickle("./data/{}-relavent.pkl".format(subreddit_name))
    print('finished download :)...')


parser = argparse.ArgumentParser(description='Scrape and preprocess reddit data.')
parser.add_argument('-n', action='store', dest='name', help='stores subreddit name')
parser.add_argument('-b', action='store', dest='begin', help='stores begining date (utc time)', default=calendar.timegm(datetime.datetime(2018, 12, 1).utctimetuple()), type=int)
parser.add_argument('-e', action='store', dest='end', help='stores ending date (utc time)', default=calendar.timegm(datetime.datetime(2019, 12, 1).utctimetuple()), type=int)
args = parser.parse_args()
main(args.name, args.begin, args.end)