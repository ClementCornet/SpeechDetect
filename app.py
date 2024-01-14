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

    st.write('Pomme/Pasteque/Carotte')

    audios = utils.get_audios(type='veg')
    train_audios = [a for a in audios[0] if 'poireau' not in a]
    train_labels = [a for a in audios[1] if 'poireau' not in a]
    
    mfccs = [
        features.extract_mfcc(audio).T for audio in train_audios
    ]

    

    audio_bytes = audio_recorder(sample_rate=22050)
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        with open('temp.wav', 'wb') as f:
            f.write(audio_bytes)
        mfcc_test = features.extract_mfcc("temp.wav").T
        

        ddd = [np.sum(features.dtw_distance(mfcc_test, mfccs[i])) for i in range(len(train_audios))]
        st.write('PREDICTED : ', train_labels[np.argmin(ddd)])
        
        os.remove('temp.wav')