import uuid, wave, numpy, os  
from datetime import datetime

from flask import Flask, current_app, session, url_for, render_template
from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
#folder where record files stored 
app.config['FILEDIR'] = 'static/_files/'
socketio = SocketIO(app)
db = SQLAlchemy(app)
frames=[]

#connect to mysql DB 'oti'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:ur_mysql_password@localhost/oti'
app.config["DEBUG"] = True
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

#record_data table(id,record,user_id,date) 
class Record_data(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    record = db.Column(db.LargeBinary(length=(2**32)-1))
    user_id=db.Column(db.Integer)
    date=db.Column(db.DateTime, nullable=False,
        default=datetime.utcnow())
    

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
    wf.setframerate(options.get('fps', 44100))
    session['wavefile'] = wf


@socketio.on('write-audio', namespace='/audio')
def write_audio(data):
    """Write a chunk of audio from the client."""
    session['wavefile'].writeframes(data)
    #print(type(data))
    frames.append(data)


@socketio.on('end-recording', namespace='/audio')
def end_recording():
    """Stop recording audio from the client."""
    emit('add-wavefile', url_for('static',
                                 filename='_files/' + session['wavename']))
    
    #Insert record as binary to DB
    record1=Record_data(record=bytes(numpy.asarray(frames)),user_id=3)
    db.session.add(record1)
    db.session.commit()
    # #read from DB
    # record2=Record_data.query.filter_by(user_id=3).first()
    # v=record2.record
    # #test
    # wr = wave.open('test3.wav', 'wb')
    # wr.setnchannels(1)
    # wr.setsampwidth(session['wavefile'].getsampwidth())
    # wr.setframerate(44100)
    # wr.writeframes(v)

    del session['wavefile']
    del session['wavename']

if __name__ == '__main__':
	socketio.run(app, debug= True)
