'''
Example of usage:

files = ["test_audios/test_cat_1.wav"]
df_raw , df_bands, df_bands_nolin, df_pitches = feature_extract(files)
'''

import librosa
import numpy as np
import pandas as pd

#our function defined at 0
#function = 1 at x=1
def function(x):
    y = -(x-1)**2 + 1
    return y
    
def feature_calc(time_feat):
    return np.mean(time_feat)

def feature_extract(files,nbands=25):
    #for ease of coding, we'll create df and then return the only row there is
    df_base = pd.DataFrame(columns = ["file","animal"])
    df_base["file"] = files
    df_base = df_base.set_index('file')
    
    for file in files:
        #get the animal
        if "dog_" in file:
            df_base.loc[file,"animal"] = 0
        elif "cat_" in file:
            df_base.loc[file,"animal"] = 1

    #create dataframess
    df_raw = df_base.copy()
    df_bands = df_base.copy()
    df_bands_nolin = df_base.copy()
    df_pitches = df_base.copy()

    #These values are a constant really but we are just gonna get them by analyzing the first file each tima
    #kinda inefficient but easy
    x, sr = librosa.load(files[0])
    chroma_stft = librosa.feature.chroma_stft(y=x, sr=sr,n_chroma=12, n_fft=500)
    X = librosa.stft(x,n_fft=500)

    #create columns for df
    #raw df
    for i in range(len(X)):
        df_raw['freq'+str(i)] = np.nan
    
    #pitch df
        for i in range(len(chroma_stft)):
            df_pitches['chroma_'+str(i)] = np.nan
        
    #bands lin
    for i in range(nbands):
        df_bands['band'+str(i)] = np.nan
        df_bands_nolin['band'+str(i)] = np.nan


    for file in files:
        #load file
        x, sr = librosa.load(file)

        #first step, get the parts where the animal is making sound
        #get the pitch data
        chroma_stft = librosa.feature.chroma_stft(y=x, sr=sr,n_chroma=12, n_fft=500)
        X = librosa.stft(x,n_fft=500)
        Xdb = librosa.amplitude_to_db(abs(X))
        
        #get the activation counts
        activation_count = np.array([0]*len(chroma_stft[0]))
        thr = 0.5
        median = [np.median(chroma_stft[i]) for i in range(len(chroma_stft))]
        for i in range(len(chroma_stft[0])):
            for pitch in range(len(chroma_stft)):
                if(chroma_stft[pitch][i] <= median[pitch]*thr):
                    activation_count[i] += 1

        count_thr=3
        #these are the points on chroma points where the count is over threshold
        chroma_points = np.where(activation_count > count_thr)[0]
        #lenght of chroma analysis audio
        chr_len = len(chroma_stft[0])


        #-----
        #Now that we have our cut references we can start by getting data for each df
        #------

        #CHROMA PITCH FF
        #let's start with what we already have cut
        for i in range(len(chroma_stft)):
            #for each chroma add points selected by activation
            df_pitches.loc[file,'chroma_'+str(i)] = feature_calc(chroma_stft[i][chroma_points])
        
        # RAW FREQ DF
        Xdb_len = len(Xdb[0]) #as is a matrix of (featuresx[samples])
        phi = int(Xdb_len / chr_len)

        #to get our slices back we will directly append it to a new variable
        Xdb_cut = [np.nan]*len(Xdb)
        for ch_idx in chroma_points:
            #reference of center of slice in x points or "coordinates"
            Xdb_idx = ch_idx * phi
            #to get the whole slice get half and half around center, evade negative indexes
            slice_0 = Xdb_idx-int(phi/2)
            if slice_0 < 0 :
                slice_0 = 0
            slice_1 = Xdb_idx+int(phi/2)
            if slice_1 < 0:
                slice_1 = 0
            #slices_ref.append([freqraw_idx-int(phi/2),freqraw_idx+int(phi/2)])
            for a in range(len(Xdb)):
                Xdb_cut[a] = np.append(Xdb_cut[a],Xdb[a][slice_0:slice_1])

        for i in range(len(Xdb_cut)):
            #for each chroma add points selected by activation
            #[1:] to avoid the first nan (little trick)
            df_raw.loc[file,'freq'+str(i)] = feature_calc(Xdb_cut[i][1:])
        
        #FREQ BANDS LINEAR DF
        nfreq = len(Xdb)
        Xdb_bnds = np.empty((nbands,len(Xdb_cut[0])))

        lamb = int(nfreq/(nbands))
        #iterate all bands we wanna create
        for band in range(nbands):
            #iterate all data points, for each calculate the mean over all frequencies inside the band
            cut1 = band*lamb
            cut2 = band*lamb+lamb 
            for data_idx in range(len(Xdb_cut[0])):
                values = []
                #retrieve points for the mean
                
                for freq in range(cut1,cut2):
                    values.append(Xdb_cut[freq][data_idx])
                
                #compute mean and append to that band
                Xdb_bnds[band][data_idx] = np.mean(values)
        
        for i in range(len(Xdb_bnds)):
            #for each chroma add points selected by activation
            #[1:] to avoid the first nan (little trick)
            df_bands.loc[file,'band'+str(i)] = feature_calc(Xdb_bnds[i][1:])
        
        #FREQ BANDS NON-LINEAR DF
        #commented because already calculated before
        #nfreq = len(Xdb)
        #frequencies = librosa.fft_frequencies(sr=sr, n_fft=500)
        Xdb_bnds = np.empty((nbands,len(Xdb_cut[0])))

        lamb = int(nfreq/nbands)
        #iterate all bands we wanna create
        for band in range(nbands):
            #iterate all data points, for each calculate the mean over all frequencies inside the band
            for data_idx in range(len(Xdb_cut[0])):
                values = []
                #retrieve points for the mean

                #run indexes through a log function to make it non linear
                #scale values to fit well log function divide by
                scalar1 = function(band/nbands)
                freq1 = int(scalar1*lamb*nbands)
                scalar2 = function((band + 1)/nbands)
                freq2 = int(scalar2*lamb*nbands)

                for freq in range(freq1,freq2):
                    values.append(Xdb_cut[freq][data_idx])
                #compute mean and append to that band
                Xdb_bnds[band][data_idx] = np.mean(values)
        
        for i in range(len(Xdb_bnds)):
            #for each chroma add points selected by activation
            #[1:] to avoid the first nan (little trick)
            df_bands_nolin.loc[file,'band'+str(i)] = feature_calc(Xdb_bnds[i][1:])

    return df_raw , df_bands, df_bands_nolin, df_pitches