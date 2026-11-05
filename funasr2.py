from datetime import datetime, timezone
from os.path import splitext, basename
from os import system, rename, remove
from time import time
from threading import Thread
from colorama import Fore, init, Style

# if splitext(video_path)[1] != 'wav':
#     path1 = video_path.replace(' ', '_')
#     rename(video_path, path1)
#     path1 = splitext(path1)[0]+'.wav'
#     system(f'ffmpeg -i {video_path} {path1}')
#     video_path = path1
video = []


def deal_input():
    while 1:
        video.append(input())


def generate(video_path):
    ouput_path = splitext(video_path)[0]+'.srt'
    raw = splitext(video_path)[0]+'_fun.txt'
    t0 = time()
    res = model.generate(input=video_path,
                         batch_size_s=300,
                         hotword='雨纪 ameki michiya 米汀 米米 维阿 罐罐 百舰 舰长 点酥 酥酥 海海 栞栞 岁己 弥希 真真 小岁'
                         )
    text = res[0]['text'].split()
    ts = res[0]['timestamp']
    # print(f'ASR:{time()-t0:.2f}s')

    res = punc.generate(input=res[0]['text'])[0]['text']
    # print(f'PUNC:{time()-t0:.2f}s')

    with open(raw, 'w', encoding='utf-8') as f:
        f.write(res.replace(' ','\n'))

    with open(ouput_path, 'w', encoding='utf-8') as f:
        n = len(ts)
        m = len(res)
        cnt = 1
        s = ""
        i = 0
        j = 0
        while i < n:
            f.write(f'{cnt:d}\n')
            dt = datetime.fromtimestamp(ts[i][0]/1000, timezone.utc)
            f.write(dt.strftime("%H:%M:%S,%f")+' --> ')
            while res[j:j+len(text[i])].lower() == text[i].lower():
                if len(s) > 12:
                    j -= 1
                    break
                s += text[i]
                j += len(text[i])
                i += 1
                if i >= n or j+len(text[i]) > m:
                    break
            if s == "":
                pass
            dt = datetime.fromtimestamp(ts[i-1][1]/1000, timezone.utc)
            f.write(dt.strftime("%H:%M:%S,%f")+'\n')
            f.write(s+'\n\n')
            s = ""
            j += 1
            cnt += 1

    print(f'Convert:{time()-t0:.2f}s')
    system(f'"{ouput_path}"')


if __name__ == '__main__':
    init()
    t1 = Thread(target=deal_input)
    t1.start()
    t0 = time()
    from funasr import AutoModel
    model = AutoModel(model="paraformer-zh",
                      vad_model="fsmn-vad",
                      #   punc_model="ct-punc",
                      #   spk_model="cam++"
                      )
    punc = AutoModel(model="ct-punc")
    print(f'Load:{time()-t0:.2f}s')
    while 1:
        if len(video):
            video_path = video.pop()
            try:
                generate(video_path)
            except BaseException as e:
                print(Fore.RED+f'ERROR:{e}'+Style.RESET_ALL)
