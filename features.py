import librosa

def extract_mfcc(filepath):
    """
    Extract MFCCs for a given .wav file
    Return 20 MFCCs
    """
    y, sr = librosa.load(filepath)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfccs.T

def extract_lpc(filepath):
    """
    Extract LPC for a given .wav file
    LPC at order 3, using Burg's method
    """
    y, sr = librosa.load(filepath)
    return librosa.lpc(y, order=3)


def dtw_distance(template, test):
    """
    Get DTW distance between 2 audio sequences.
    Using the euclidean distance as a local distance
    """

    D, _ = librosa.sequence.dtw(template, test, metric='minkowski') # Matrice de coût pour la DTW

    normalized_distance = D[-1, -1] / sum(D.shape) # Distance DTW normalisée
    return normalized_distance