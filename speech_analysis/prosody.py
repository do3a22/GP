import parselmouth
from parselmouth.praat import call
import numpy as np
from report_dict import voice_report_dict

def extract_prosodic(audio_file, start_sec = 0.0, end_sec = 0.0):
    features = np.array([])

    snd = parselmouth.Sound(audio_file)
    #snd = call(snd, "Remove noise", 0, 0, 0.025, 80, 10000, 40, "Spectral subtraction")
    duration = snd.get_total_duration()

    spec = snd.to_spectrum()
    energy = spec.get_band_energy(0, 10000)
    features = np.append(features, energy)

    pitch = call(snd, "To Pitch", 0, 75, 600)
    f0mean = call(pitch, "Get mean", start_sec, end_sec, "Hertz")
    f0min = call(pitch, "Get minimum", start_sec, end_sec, "Hertz", "Parabolic")
    f0max = call(pitch, "Get maximum", start_sec, end_sec, "Hertz", "Parabolic")
    f0range = f0max - f0min
    f0sd = call(pitch, "Get standard deviation", start_sec, end_sec, "Hertz")
    features = np.append(features, [f0mean, f0min, f0max, f0range, f0sd])

    intensity = call(snd, "To Intensity", 50, 0, "yes")
    intens_min = call(intensity, "Get minimum", start_sec, end_sec, "Parabolic")
    intens_max = call(intensity, "Get maximum", start_sec, end_sec, "Parabolic")
    intense_sd = call(intensity, "Get standard deviation", start_sec, end_sec)
    intense_mean = call(intensity, "Get mean", start_sec, end_sec, "energy")
    intens_range = intens_max - intens_min
    features = np.append(features, [intense_mean, intens_min, intens_max, intens_range, intense_sd])

    formant = call(snd, "To Formant (burg)", 0, 3, 5500, 0.025, 50)
    f1mean = call(formant, "Get mean", 1, 0, 0, "hertz")
    f2mean = call(formant, "Get mean", 2, 0, 0, "hertz")
    f3mean = call(formant, "Get mean", 3, 0, 0, "hertz")
    f1sd = call(formant, "Get standard deviation", 1, 0, 0, "hertz")
    f2sd = call(formant, "Get standard deviation", 2, 0, 0, "hertz")
    f3sd = call(formant, "Get standard deviation", 3, 0, 0, "hertz")
    f2meanf1 = f2mean / f1mean
    f3meanf1 = f3mean / f1mean
    f2sdf1 = f2sd / f1sd
    f3sdf1 = f3sd / f1sd
    f1bw = call(formant, "Get maximum", 1, 0, 0, "hertz", "Parabolic")
    f2min = call(formant, "Get minimum", 2, 0, 0, "hertz", "Parabolic")
    f2max = call(formant, "Get maximum", 2, 0, 0, "hertz", "Parabolic")
    f2bw = f2max - f2min
    f3min = call(formant, "Get minimum", 3, 0, 0, "hertz", "Parabolic")
    f3max = call(formant, "Get maximum", 3, 0, 0, "hertz", "Parabolic")
    f3bw = f3max - f3min
    features = np.append(features, [f1mean, f2mean, f3mean, f1sd, f2sd, f3sd, f1bw, f2bw, f3bw, f2meanf1, f3meanf1, f2sdf1, f3sdf1])

    pp = call(snd, "To PointProcess (periodic, cc)", 50, 600)
    voice_report_str = parselmouth.praat.call([snd, pitch, pp], "Voice report", 0.0, 0.0, 50, 600, 1.3, 1.6, 0.03, 0.45)
    rep_dict = voice_report_dict(voice_report_str)
    jitter = rep_dict['Jitter (local)']
    shimmer = rep_dict['Shimmer (local)']
    features = np.append(features, [jitter, shimmer])

    breaks = rep_dict['Degree of voice breaks']
    unvoiced = rep_dict['Fraction of locally unvoiced frames']
    features = np.append(features, [duration, unvoiced, breaks])

    allDurPause = 0
    avgDurPause = 0
    maxDurPause = 0
    count = 0
    txtGrid = call(intensity, "To TextGrid (silences)", -25, 0.7, 0.032, "silent", "sounding")
    total = call(txtGrid, "Get number of intervals", 1)
    for i in range(total):
        typ = call(txtGrid, "Get label of interval", 1, i+1)
        if (typ == "silent"):
            st = call(txtGrid, "Get starting point", 1, i+1)
            en = call(txtGrid, "Get end point", 1, i+1)
            if (maxDurPause < en-st):
                maxDurPause = en-st
            count = count + 1
            allDurPause = allDurPause + en-st
    avgDurPause = allDurPause / count
    features = np.append(features, [maxDurPause, avgDurPause, allDurPause])

    return features

print(extract_prosodic("P1.wav"))
