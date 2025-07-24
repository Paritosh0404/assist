import pyttsx3, queue, threading, time
engine = pyttsx3.init()
engine.setProperty('rate', 170)
_q = queue.Queue()

def _worker():
    while True:
        txt = _q.get()
        engine.say(txt)
        engine.runAndWait()
threading.Thread(target=_worker, daemon=True).start()

def speak(txt):
    if _q.qsize() < 3:      # simple flood control
        _q.put(txt)
