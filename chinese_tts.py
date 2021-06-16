import numpy as np
import soundfile as sf
import yaml

import tensorflow as tf

from tensorflow_tts.inference import AutoProcessor
from tensorflow_tts.inference import TFAutoModel

processor = AutoProcessor.from_pretrained("tensorspeech/tts-fastspeech2-baker-ch")
fastspeech2 = TFAutoModel.from_pretrained("tensorspeech/tts-fastspeech2-baker-ch")
from tensorflow_tts.inference import TFAutoModel

model = TFAutoModel.from_pretrained("tensorspeech/tts-mb_melgan-baker-ch")

text = "这是一个开源的端到端中文语音合成系统"

input_ids = processor.text_to_sequence(text, inference=True)

mel_before, mel_after, duration_outputs, _, _ = fastspeech2.inference(
    input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
    speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
    speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
    f0_ratios =tf.convert_to_tensor([1.0], dtype=tf.float32),
    energy_ratios =tf.convert_to_tensor([1.0], dtype=tf.float32),
)

audios = model.inference(mel_after)[0, :, 0]
print(1)



# save to file
import time
a=time.time()
import time
now = time.localtime()
nowt = time.strftime("%Y-%m-%d-%H-%M-%S", now)  #这一步就是对时间进行格式化
print(nowt)
sf.write(f'./{nowt}.wav', audios, 22050, 'PCM_24')