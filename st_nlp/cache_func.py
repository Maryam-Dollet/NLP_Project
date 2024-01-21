import streamlit as st
import pandas as pd
import json
from collections import defaultdict  # For word frequency
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec
from gensim.models import LdaModel
from gensim import corpora
import nltk
from nltk.tokenize import word_tokenize
from nltk import ngrams
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from hdbscan import HDBSCAN
import pyLDAvis
import pyLDAvis.gensim
from wordcloud import WordCloud

# Load Data
@st.cache_data
def load_companies_translated():
    return pd.read_csv("data/company_desc_translated.csv", sep=";").dropna(ignore_index=True)


@st.cache_data
def load_reviews_sample():
    return pd.read_csv("data/reviews_sample_translated.csv", sep=";").dropna(ignore_index=True)


@st.cache_data
def load_category():
    return pd.read_csv("data/category_data.csv", sep=";").dropna(ignore_index=True)


@st.cache_data
def load_reviews_sample2():
    return pd.read_csv("data/reviews_sample.csv")

# Embeddings
@st.cache_data
def load_model(path: str):
    model = Word2Vec.load(path)
    return model


def get_similarity(model, token):
    return model.wv.most_similar(positive=[token])


@st.cache_data
def get_PCA(_model):
    labels = list(_model.wv.key_to_index.keys())
    vectors = _model.wv[_model.wv.key_to_index.keys()]

    pca = PCA(n_components=3)
    result = pca.fit_transform(vectors)

    result_df = pd.DataFrame(result, columns=["x", "y", "z"])
    result_df["word"] = labels
    return result_df


@st.cache_data
def get_TSNE(_model):
    labels = list(_model.wv.key_to_index.keys())
    vectors = _model.wv[_model.wv.key_to_index.keys()]

    tsne = TSNE(n_components=3, verbose=1,n_iter=1000,random_state=1)
    tsne_results = tsne.fit_transform(vectors)

    result_df = pd.DataFrame(tsne_results, columns=["x", "y", "z"])
    result_df["word"] = labels

    return result_df


@st.cache_data
def get_UMAP(_model):
    labels = list(_model.wv.key_to_index.keys())
    vectors = _model.wv[_model.wv.key_to_index.keys()]

    umap_3d = UMAP(n_components=3, init='random', random_state=0)
    proj_3d = umap_3d.fit_transform(vectors)

    result_df = pd.DataFrame(proj_3d, columns=["x", "y", "z"])
    result_df["word"] = labels

    return result_df

# Clustering
def hdbscan_cluster(df):
    clusterable_embedding = list(df[["x", "y", "z"]].values)
    labels = HDBSCAN(min_samples=10,min_cluster_size=20,).fit_predict(clusterable_embedding)
    df["category"] = labels
    df["category"] = df["category"].astype(str)
    return df

# Semantic Search functions
@st.cache_data
def load_company_tagged():
    return pd.read_csv("data/company_tagged.csv", sep=";")


@st.cache_data
def load_doc2vec():
    return Doc2Vec.load("models/d2v.model")


def find_similar_doc(_model, sentence: str, company_df):
    test_data = word_tokenize(sentence.lower())
    v1 = _model.infer_vector(test_data)

    sims = _model.dv.most_similar([v1])
    best_match = [x[0] for x in sims]

    return best_match

#Topic Modeling
@st.cache_data
def load_corpus():
    with open("data/pipe7.json", 'r') as f:
        data = json.load(f)
    return data


@st.cache_data
def LDA(corpus, nb_topics):
    dictionary = corpora.Dictionary(corpus)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in corpus]

    # LSA model
    lda = LdaModel(doc_term_matrix, num_topics=nb_topics, id2word = dictionary)

    vis = pyLDAvis.gensim.prepare(lda, doc_term_matrix, dictionary)

    html_str = pyLDAvis.prepared_data_to_html(vis)

    n_words = 10

    topic_words = pd.DataFrame({})

    for i, topic in enumerate(lda.get_topics()):
        top_feature_ids = topic.argsort()[-n_words:][::-1]
        feature_values = topic[top_feature_ids]
        words = [dictionary[id] for id in top_feature_ids]
        topic_df = pd.DataFrame({'value': feature_values, 'word': words, 'topic': i})
        topic_words = pd.concat([topic_words, topic_df], ignore_index=True)

    # LSA model
    return lda, html_str, topic_words


@st.cache_data
def get_freq(desc_token):
    freq = {}
    for desc in desc_token:
        for item in desc:
            if (item in freq):
                freq[item] += 1
            else:
                freq[item] = 1
    return freq


@st.cache_data
def get_wordcloud(freq):
    return WordCloud().fit_words(freq)


# we create a function that takes the number of words we want in the n-grams as an argument, doing the same thing than before
@st.cache_data
def get_ngrams(n, company_df, pipe7):
    company_id = 0
    company_ngram_freq_dict = dict()
    categories = list(company_df["category"].unique())
    for company in pipe7:
        freq_dict = defaultdict(int)

        for token in ngrams(company, n): # Count token frequency in each company description
            freq_dict[token] += 1
        company_ngram_freq_dict[company_df["company_name"][company_id]] = freq_dict # Add company name as key to dict and the frequency dictionnary as value

        company_id += 1

    dict_cat = {}
    for category in categories:
        # print(category + " :")
        # we merge all the dictionaries of the companies in the same category
        merged_dict = defaultdict(int)
        for company in company_df[company_df["category"] == category]["company_name"]:
            for token in company_ngram_freq_dict[company]:
                merged_dict[token] += company_ngram_freq_dict[company][token]
        # we print the 5 most frequent bigrams in the category
        # print(sorted(merged_dict, key=merged_dict.get, reverse=True)[:5])
        dict_cat[category] = sorted(merged_dict, key=merged_dict.get, reverse=True)[:5]

    return dict_cat


@st.cache_data
def get_UMAP_d2v(_model, df):
    tags = df["tag"]
    labels = df["company_name"]
    category = df["category"]
    vectors = [_model.dv[tag] for tag in tags]
    umap_3d = UMAP(n_components=3, init='random', random_state=0)
    proj_3d = umap_3d.fit_transform(vectors)


    result_df_umap = pd.DataFrame(proj_3d, columns=["x", "y", "z"])
    result_df_umap["word"] = labels
    result_df_umap["cat"] = category

    return result_df_umap

# Clustering
def hdbscan_cluster_2(df):
    clusterable_embedding = list(df[["x", "y", "z"]].values)
    labels = HDBSCAN(min_samples=20,min_cluster_size=50,).fit_predict(clusterable_embedding)
    df["category"] = labels
    df["category"] = df["category"].astype(str)
    return df

# For Whatever reason need to install punkt for chatbot
def punkt():
    nltk.download('punkt')