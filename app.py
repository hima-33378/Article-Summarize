from flask import Flask, render_template
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_python_script', methods=['POST'])
def run_python_script():
    # Execute your Python script here
    result = subprocess.run(['python', 'jupytermain.py'], stdout=subprocess.PIPE, text=True)
    output = result.stdout
    return output

if __name__ == '__main__':
    app.run(debug=True)