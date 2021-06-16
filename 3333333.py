#https://github.com/TensorSpeech/TensorFlowTTS






from tensorflow_tts.inference import TFAutoModel

model = TFAutoModel.from_pretrained("tensorspeech/tts-mb_melgan-baker-ch")
audios = model.inference(mels)