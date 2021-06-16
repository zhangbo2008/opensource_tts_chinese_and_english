import numpy as np
import soundfile as sf
import yaml
# python -m pip install --upgrade pip
#pip install tensorflow-gpu==2.5
# pip install numba==0.48         安装nltk   https://blog.csdn.net/weixin_44633882/article/details/104494276  下载完放入 C:\nltk_data\corpora

import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"





import soundfile as sf
import numpy as np

import tensorflow as tf

from tensorflow_tts.inference import AutoProcessor
from tensorflow_tts.inference import TFAutoModel
#============需要3个模型才能输出出结果!!!!!!!!!!!!!!
processor = AutoProcessor.from_pretrained("tensorspeech/tts-tacotron2-ljspeech-en")
tacotron2 = TFAutoModel.from_pretrained("tensorspeech/tts-tacotron2-ljspeech-en")
mb_melgan = TFAutoModel.from_pretrained("tensorspeech/tts-mb_melgan-ljspeech-en")

text = "This is a demo to show how to use our model to generate mel spectrogram from raw text."

input_ids = processor.text_to_sequence(text)
# input_ids最后一个加入的是结束符.
# tacotron2 inference (text-to-mel)
decoder_output, mel_outputs, stop_token_prediction, alignment_history = tacotron2.inference(
    input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
    input_lengths=tf.convert_to_tensor([len(input_ids)], tf.int32),
    speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
)

# melgan inference (mel-to-wav)
audio = mb_melgan.inference(mel_outputs)[0, :, 0]

# save to file
import time
a=time.time()
import time
now = time.localtime()
nowt = time.strftime("%Y-%m-%d-%H-%M-%S", now)  #这一步就是对时间进行格式化
print(nowt)
sf.write(f'./{nowt}.wav', audio, 22050, "PCM_16")