import streamlit as st
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec
from nltk.tokenize import word_tokenize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from hdbscan import HDBSCAN


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

    pca = PCA(n_components=2)
    result = pca.fit_transform(vectors)

    result_df = pd.DataFrame(result, columns=["x", "y"])
    result_df["word"] = labels
    return result_df


@st.cache_data
def get_TSNE(_model):
    labels = list(_model.wv.key_to_index.keys())
    vectors = _model.wv[_model.wv.key_to_index.keys()]

    tsne = TSNE(n_components=2, verbose=1,n_iter=1000,random_state=1)
    tsne_results = tsne.fit_transform(vectors)

    result_df = pd.DataFrame(tsne_results, columns=["x", "y"])
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
