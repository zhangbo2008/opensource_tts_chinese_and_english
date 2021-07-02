import librosa
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset  # pip 一下 datasets  , soundfile
import soundfile as sf

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")


def map_to_array(batch):
    speech, _ = sf.read(batch["file"])

    batch["speech"] = speech  # 把文件放到speech的value里面
    return batch
if 0:
    ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
    ds = ds.map(map_to_array)  # map : Apply a function to all the elements in the table (individually or in batches)
    # 我们只玩句子里面第一句.
    input_values = processor(ds["speech"][0], return_tensors="pt").input_values  # Batch size 1
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)

    transcription = processor.decode(predicted_ids[0])

     # compute loss
    target_transcription = "A MAN SAID TO THE UNIVERSE SIR I EXIST"

     # wrap processor as target processor to encode labels
    with processor.as_target_processor():
      labels = processor(target_transcription, return_tensors="pt").input_ids       # 把答案也进行编码,跟语音编码是一样的.上面用的特征提取器, 下面用的nlp编码.
    # 编码之后也是38这个长度.
    loss = model(input_values, labels=labels).loss
    print(loss)


#{'<pad>': 0, '<s>': 1, '</s>': 2, '<unk>': 3, '|': 4, 'E': 5, 'T': 6, 'A': 7, 'O': 8, 'N': 9, 'I': 10, 'H': 11, 'S': 12, 'R': 13, 'D': 14, 'L': 15, 'U': 16, 'M': 17, 'W': 18, 'C': 19, 'F': 20, 'G': 21, 'Y': 22, 'P': 23, 'B': 24, 'V': 25, 'K': 26, "'": 27, 'X': 28, 'J': 29, 'Q': 30, 'Z': 31} 编码字典是这个.



# 下面在自己的数据集上使用.

from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC
from datasets import load_dataset  # 这2个库包都pip一下就行.
import soundfile as sf
import torch
from datasets import load_dataset

# load model and tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# name='2.mp3'
name='11.wav'
name='22English.wav'
if ".wav" in name:
    src_sig,sr = sf.read(name)  #name是要 输入的wav 返回 src_sig:音频数据  sr:原采样频率
    dst_sig = librosa.resample(src_sig,sr,16000)  #resample 入参三个 音频数据 原采样频率
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
logits = model(input_values).logits

# take argmax and decode
predicted_ids = torch.argmax(logits, dim=-1)
transcription = tokenizer.batch_decode(predicted_ids) # 下面看这个解码算法, 应该是vitebi.....其实不是,只是简单的拼接......因为都没给概率......
print(transcription)
