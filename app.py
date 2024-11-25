from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from neural_networks import visualize

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_experiment', methods=['POST'])
def run_experiment():
    data = request.json
    activation = data['activation']
    learning_rate = float(data['lr'])
    num_steps = int(data['step_num'])

    visualize(activation, learning_rate, num_steps)

    gif_path = "results/visualize.gif"
    return jsonify({
        "result_gif": gif_path if os.path.exists(gif_path) else None,
    })

@app.route('/results/<filename>')
def results(filename):
    return send_from_directory('results', filename)

if __name__ == '__main__':
    app.run(debug=True, port=3000)
