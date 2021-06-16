# https://www.tingclass.net/show-5056-1058-1.html

from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC
from datasets import load_dataset  # 这2个库包都pip一下就行.
import soundfile as sf
import torch
from datasets import load_dataset
librispeech_eval = load_dataset("librispeech_asr", "clean", split="test")
# load model and tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
tokenizer.save_vocabulary("tmp")
# define function to read in sound file
# def map_to_array(batch):
#     speech, _ = sf.read(batch["file"])
#     batch["speech"] = speech
#     return batch
#
# # load dummy dataset and read soundfiles
# ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
# ds = ds.map(map_to_array)
#
#
#
#
# # custome data
# # ds["speech"]=sf.read('11.wav')[0]
# #
#
# #token
# # tokenize
# input_values = tokenizer(ds["speech"][:2], return_tensors="pt", padding="longest").input_values  # Batch size 1


import librosa
import soundfile as sf


# name='2.mp3'
name='11.wav'
if ".wav" in name:
    src_sig,sr = sf.read(name)  #name是要 输入的wav 返回 src_sig:音频数据  sr:原采样频率
    dst_sig = librosa.resample(src_sig,sr,16000)  #resample 入参三个 音频数据 原采样频率 和目标采样频率
# if ".mp3" in name :
#     from pydub import AudioSegment
#
#     sound = AudioSegment.from_file(name, format='MP3')
#     MP3_File = AudioSegment.from_mp3(file=name)
#     MP3_File.export('tmp.wav', format="wav")
#
#     src_sig,sr = sf.read('tmp.wav')  #name是要 输入的wav 返回 src_sig:音频数据  sr:原采样频率
#     dst_sig = librosa.resample(src_sig,sr,16000)  #resample 入参三个 音频数据 原采样频率

# 基本啥也没干的tokenizer
input_values = tokenizer(dst_sig, return_tensors="pt", padding="longest").input_values  # Batch size 1

# retrieve logits
logits = model(input_values,labels=['after']).logits

# take argmax and decode
predicted_ids = torch.argmax(logits, dim=-1)
transcription = tokenizer.batch_decode(predicted_ids) # 下面看这个解码算法, 应该是vitebi.....其实不是,只是简单的拼接......因为都没给概率......
print(transcription)


