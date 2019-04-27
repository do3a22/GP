import uuid, wave, numpy, os,time
import scipy.io.wavfile as wav
import speech_recognition as sr
from datetime import datetime
from flask import Flask, current_app, session, url_for, render_template
from flask_socketio import SocketIO, emit
#from flask_sqlalchemy import SQLAlchemy
from predictVoice import tts


app = Flask(__name__)
#path for recorded files 
app.config['FILEDIR'] = 'static/_files/'
socketio = SocketIO(app)
#db = SQLAlchemy(app)
transcript=""
frames = []

#connect to mysql DB 'oti'
# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:1235789@localhost/oti'
# app.config["DEBUG"] = True
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

# #record_data table(id,record,user_id,date) 
# class Record_data(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     record = db.Column(db.LargeBinary(length=(2**32)-1))
#     user_id=db.Column(db.Integer)
#     date=db.Column(db.DateTime, nullable=False,
#         default=datetime.utcnow())
    

@app.route('/')
def index():
    """Return the client application."""
    return render_template('main.html')

@socketio.on('start-recording', namespace='/audio')
def start_recording(options):
    """Start recording audio from the client."""
    id = uuid.uuid4().hex  # server-side filename
    session['wavename'] = id + '.wav'
    wf = wave.open(current_app.config['FILEDIR'] + session['wavename'], 'wb')
    wf.setnchannels(options.get('numChannels', 1))
    wf.setsampwidth(options.get('bps', 16) // 8)
    wf.setframerate(options.get('fps', 16000))
    session['wavefile'] = wf


@socketio.on('write-audio', namespace='/audio')
def write_audio(data):
    """Write a chunk of audio from the client."""
    session['wavefile'].writeframes(data)
    frames.append(data)


 
@socketio.on('end-recording', namespace='/audio')
def end_recording():
    """Stop recording audio from the client."""
    emit('add-wavefile', url_for('static',
                                 filename='_files/' + session['wavename']))
    
    #Insert record as binary to DB
    # record1=Record_data(record=bytes(numpy.asarray(session['frame'])),user_id=3)
    # db.session.add(record1)
    # db.session.commit()

    #convert speech to text
    filepath=app.config['FILEDIR']+session['wavename']
    recog = sr.Recognizer()
    audioFile = sr.AudioFile(filepath)
    session['transcript'] = transcript
    with audioFile as source:
        audio = recog.listen(source)
        session['transcript'] = session['transcript'] + " " + recog.recognize_google(audio)
    emit('textDone',session['transcript'])
    del session['wavefile']
    del session['wavename']
    

 #TTS
#@socketio.on('TextToSpeech', namespace='/audio')
#def TextToSpeech():
#    out_path=tts(session['transcript'])
#    print(out_path)
#    emit('TTS',out_path)   
#    del session['transcript']
  
    
@socketio.on('TextToSpeech', namespace='/audio')
def TextToSpeech():
    file_name=tts(session['transcript'])
    print(file_name)
    emit('TTS',url_for('static',
                                 filename='samples/' + file_name))   
    del session['transcript']


    
if __name__ == '__main__':
	socketio.run(app,port=5700,debug= True)
  