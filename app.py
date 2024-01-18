import streamlit as st
#import plotly.express as px
import pandas as pd
#from sklearn.manifold import MDS
import numpy as np

import utils
import features

from audio_recorder_streamlit import audio_recorder
import os

from hmmlearn import hmm



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
    

    model_coco = hmm.GMMHMM(n_components=2, n_mix=1)
    model_cerise = hmm.GMMHMM(n_components=2, n_mix=1)
    model_banane = hmm.GMMHMM(n_components=2, n_mix=1)

    train_audios, train_labels = utils.get_audios(type='veg2')
    for audio in train_audios:
        mfcc = features.extract_mfcc(audio)
        if 'coco' in audio:
            model_coco.fit(mfcc)
        elif 'banane' in audio:
            model_banane.fit(mfcc)
        elif 'cerise' in audio:
            model_cerise.fit(mfcc)

    audio_bytes = audio_recorder(sample_rate=22050)
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        with open('temp.wav', 'wb') as f:
            f.write(audio_bytes)
        mfcc_test = features.extract_mfcc("temp.wav").T

        lpc_test = features.extract_lpc("temp.wav")
        #st.write(lpc_test)
        

        ddd = [np.sum(features.dtw_distance(mfcc_test, mfccs[i])) for i in range(len(train_audios))]
        lll = [np.linalg.norm(lpc_test - lpcs[i], ord=2) for i in range(len(train_audios))]
        st.write('Prediction with MFCC: ', train_labels[np.argmin(ddd)])
        st.write('Prediction with LPC : ', train_labels[np.argmin(lll)])

        df = pd.DataFrame()
        df.index = [t.split('\\')[-1] for t in train_audios]
        df['lpc_dist'] = lll
        df['mfcc_dtw'] = ddd
        #st.write(df.sort_values('lpc_dist', ascending=True).style.highlight_min())

        

        score_coco, score_banane, score_cerise = model_coco.score(mfcc_test.T), model_banane.score(mfcc_test.T), model_cerise.score(mfcc_test.T)
        
        #st.write(f"coco {score_coco}, banane {score_banane}, cerise {score_cerise}")
        st.write(f"Prediction with GMMHMM: {['coco', 'banane', 'cerise'][np.argmax([score_coco, score_banane, score_cerise])]}")
        #st.write(f"GMMHMM PREDICTION MIN: {['coco', 'banane', 'cerise'][np.argmin([score_coco, score_banane, score_cerise])]}")

        os.remove('temp.wav')