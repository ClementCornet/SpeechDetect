import streamlit as st
#import plotly.express as px
import pandas as pd
#from sklearn.manifold import MDS
import numpy as np

import utils
import features

from audio_recorder_streamlit import audio_recorder
import os



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
        st.write('PREDICTED MFCC: ', train_labels[np.argmin(ddd)])
        st.write('PREDICTED LPC : ', train_labels[np.argmin(lll)])

        df = pd.DataFrame()
        df.index = [t.split('\\')[-1] for t in train_audios]
        df['lpc_dist'] = lll
        df['mfcc_dtw'] = ddd
        st.write(df.sort_values('lpc_dist', ascending=True).style.highlight_min())

        os.remove('temp.wav')