from models import x_asr
from realtime_speech import start_transcribe

start_transcribe(x_asr("1920"), need_punc=False, timing=True)
