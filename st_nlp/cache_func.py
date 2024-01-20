import streamlit as st
import pandas as pd
from gensim.models import Word2Vec
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


@st.cache_data
def load_w2v():
    w2v_model = Word2Vec.load('models/w2v_company_desc_model')
    return w2v_model


def get_similarity(model, token):
    return model.wv.most_similar(positive=[token])


@st.cache_data
def load_glove():
    w2v_model = Word2Vec.load('models/glove_transfer')
    return w2v_model


@st.cache_data
def load_doc2vec():
    w2v_model = Word2Vec.load('models/d2v.model')
    return w2v_model


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