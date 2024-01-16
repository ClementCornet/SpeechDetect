import streamlit as st
#import plotly.express as px
import pandas as pd
#from sklearn.manifold import MDS
import numpy as np

import utils
import features

from sklearn.manifold import MDS
import plotly.express as px


if __name__ == '__main__':
    st.write('Coco/Banane/Cerise')

    audios = utils.get_audios(type='veg2')
    train_audios = [a for a in audios[0] if 'pomme' not in a]
    train_labels = [a for a in audios[1] if 'pomme' not in a]
    
    mfccs = [
        features.extract_mfcc(audio).T for audio in train_audios
    ]

    lpcs = [
        features.extract_lpc(audio) for audio in train_audios
    ]


    dtw_mfccs = [[np.sum(features.dtw_distance(mfccs[j], mfccs[i])) for i in range(len(train_audios))] for j in range(len(train_audios))]
    

    st.dataframe(pd.DataFrame(dtw_mfccs))
    mds = MDS(n_components=2, dissimilarity='precomputed')
    emb = mds.fit_transform(dtw_mfccs)
    emb = pd.DataFrame(emb)
    
    fig = px.scatter(emb, 0,1, color=train_labels)
    st.plotly_chart(fig)


    dtw_lpcs = [[np.sum(features.dtw_distance(lpcs[j], lpcs[i])) for i in range(len(train_audios))] for j in range(len(train_audios))]
    

    st.dataframe(pd.DataFrame(dtw_lpcs))
    mds = MDS(n_components=2, dissimilarity='precomputed')
    emb = mds.fit_transform(dtw_lpcs)
    emb = pd.DataFrame(emb)
    
    fig = px.scatter(emb, 0,1, color=train_labels)
    st.plotly_chart(fig)