import torch
import numpy as np
import librosa
import config
import time
from time import sleep

from audio_buffer import AudioBuffer
from model.htsat import HTSAT_Swin_Transformer

class Audio_Classification:
    def __init__(self, model_path, config = config):
        super().__init__()

        self.device = None
        self.sed_model = HTSAT_Swin_Transformer(
            spec_size=config.htsat_spec_size,
            patch_size=config.htsat_patch_size,
            in_chans=1,
            num_classes=config.classes_num,
            window_size=config.htsat_window_size,
            config = config,
            depths = config.htsat_depth,
            embed_dim = config.htsat_dim,
            patch_stride=config.htsat_stride,
            num_heads=config.htsat_num_head
        )
        ckpt = torch.load(model_path, map_location="cpu")
        temp_ckpt = {}
        for key in ckpt["state_dict"]:
            temp_ckpt[key[10:]] = ckpt['state_dict'][key]
        self.sed_model.load_state_dict(temp_ckpt)
        self.sed_model.to(self.device)
        self.sed_model.eval()


    def predict(self, audiofile):

        if type(audiofile) == np.ndarray:
            waveform = audiofile
        else:
            waveform, sr = librosa.load(audiofile, sr=32000)

        with torch.no_grad():
            x = torch.from_numpy(waveform).float().to(self.device)
            output_dict = self.sed_model(x[None, :], None, True)
            pred = output_dict['clipwise_output']
            pred_post = pred[0].detach().cpu().numpy()
            pred_label = np.argmax(pred_post)
            pred_prob = np.max(pred_post)
        return pred_label, pred_prob


if __name__ == "__main__":
    
    # infer the single data to check the result
    # get a model you saved
    model_path = "model-weights.ckpt"
    Audiocls = Audio_Classification(model_path)

    audio_buffer = AudioBuffer(seconds=5)
    audio_buffer.start()
    sleep(5)

    while True:
        data = audio_buffer()
        print(type(data))
        print(data.shape)
        
        # Inference
        start_time = time.time()
        pred_label, pred_prob = Audiocls.predict(data)
        processing_time = time.time() - start_time

        if pred_label == 0:
            print("Barking Dog detected!")
        print('Audiocls predict output: ', pred_label, pred_prob)
        print('Processing time: ', processing_time)

        sleep(4.5 - processing_time)


    """ fig = plt.figure()
    ax = fig.add_subplot(111)
    amp = 10000 # you might need to adjust this parameter
    line, = ax.plot(amp * np.random.random(len(audio_buffer)) - amp/2)
    def animate(i):
        data = audio_buffer()
        line.set_ydata(data)
        return (line,)
    
    anim = animation.FuncAnimation(fig, animate, interval=1, blit=True)
    plt.show() """