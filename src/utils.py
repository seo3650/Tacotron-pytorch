import librosa
import librosa.filters
import numpy as np
from scipy import signal
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os.path import join
import soundfile as sf

import lws

class AudioProcessor(object):
    """A class to propress audio. Adapted from keithito: "https://github.com/keithito/tacotron/blob/master/util/audio.py"
    """
    def __init__(self, sample_rate, num_mels, num_freq, frame_length_ms, frame_shift_ms, preemphasis,
            min_level_db, ref_level_db, griffin_lim_iters, power, hop_length, fmin, fmax,
            rescaling, rescaling_max):
        self.sr = sample_rate
        self.n_mels = num_mels
        self.n_fft = (num_freq - 1) * 2
        self.win_length = int(frame_length_ms / 1000 * sample_rate)
        self.preemph = preemphasis
        self.min_level_db = min_level_db
        self.ref_level_db = ref_level_db
        self.GL_iter = griffin_lim_iters
        self.power = power
        self.frame_shift_ms = frame_shift_ms
        self.hop_length = hop_length
        self.hop_length = self._get_hop_length()
        self.fmax = fmax
        self.fmin = fmin
        assert self.fmax <= self.sr // 2
        self.mel_basis = librosa.filters.mel(self.sr, self.n_fft, 
                                            fmin=self.fmin, fmax=self.fmax,
                                            n_mels=self.n_mels)
        self.rescaling = rescaling
        self.rescaling_max = rescaling_max


    def load_wav(self, path):
        wav = librosa.core.load(path, sr=self.sr)[0]
        if self.rescaling:
            wav = wav / np.abs(wav).max() * self.rescaling_max
        return wav

    def save_wav(self, wav, path):
        #wav *= 32767 / max(0.01, np.max(np.abs(wav)))
        #wavfile.write(path, self.sr, wav.astype(np.int16))
        sf.write(path, wav, self.sr, subtype='PCM_16')

    def preemphasis(self, wav):
        return signal.lfilter([1, -self.preemph], [1], wav)

    def inv_preemphasis(self, wav_preemph):
        return signal.lfilter([1], [1, -self.preemph], wav_preemph)

    def spectrogram(self, wav):
        D = self._lws_processor().stft(self.preemphasis(wav)).T
        S = self._amp_to_db(np.abs(D)) - self.ref_level_db
        return self._normalize(S)

    def inv_spectrogram(self, linear_spect):
        '''Converts spectrogram to waveform using librosa'''
        S = self._db_to_amp(self._denormalize(linear_spect) + self.ref_level_db)  # Convert back to linear
        return self.inv_preemphasis(self._griffin_lim(S ** self.power))  # Reconstruct phase

    def melspectrogram(self, wav):
        D = self._lws_processor().stft(self.preemphasis(wav)).T
        S = self._amp_to_db(self._linear_to_mel(np.abs(D))) - self.ref_level_db
        return self._normalize(S)

    def _griffin_lim(self, S):
        '''librosa implementation of Griffin-Lim
        Based on https://github.com/librosa/librosa/issues/434
        '''
        S = S.T
        angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
        S_complex = np.abs(S).astype(np.complex)

        y = self._lws_processor().istft(S_complex * angles)
        for i in range(self.GL_iter):
          angles = np.exp(1j * np.angle(self._lws_processor().stft(y)))
          y = self._lws_processor().istft(S_complex * angles)
        return y

    def _stft(self, x):
        return librosa.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)

    def _istft(self, x):
        return librosa.istft(x, hop_length=self.hop_length, win_length=self.win_length)

    def _linear_to_mel(self, linear_spect):
        return np.dot(self.mel_basis, linear_spect)

    def _amp_to_db(self, x):
        min_level = np.exp(self.min_level_db / 20 * np.log(10))
        return 20 * np.log10(np.maximum(min_level, x))

    def _db_to_amp(self, x):
        return np.power(10.0, x * 0.05)

    def _normalize(self, x):
        return np.clip((x - self.min_level_db) / -self.min_level_db, 0, 1)

    def _denormalize(self, x):
        return (np.clip(x, 0, 1) * -self.min_level_db) + self.min_level_db
    
    def _lws_processor(self):
        return lws.lws(self.n_fft, self.hop_length, mode='speech')

    def _get_hop_length(self):
        hop_length = self.hop_length
        if hop_length == 'None':
            hop_length = int(self.frame_shift_ms / 1000 * self.sr)
        return hop_length


def make_spec_figure(spec, audio_processor):
    spec = audio_processor._denormalize(spec)
    fig = plt.figure(figsize=(16, 10))
    plt.imshow(spec.T, aspect="auto", origin="lower")
    plt.colorbar()
    plt.tight_layout()
    return fig


def make_attn_figure(attn):
    fig, ax = plt.subplots()
    im = ax.imshow(
        attn.T,
        aspect='auto',
        origin='lower',
        interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()
    return fig

