import logging
import os
import time
import markdown

from flask import Flask, Markup, json, jsonify, request, render_template
from utils.log_func import get_log_func
from http_utils import DataEncoder, read_json, write_json

import torch
from src.evaluation import predict_text
from src.models import initialize_inference_model, attempt_load_model

log_level = os.getenv("LOG_LEVEL", "WARNING")
root_logger = logging.getLogger()
root_logger.setLevel(log_level)
log = get_log_func(__name__)

log("starting flask app", level="debug")
app = Flask(__name__)
app.logger.debug('debug')

# dictionary mapping style indices to styles
STYLES = {
    "informal": 0,
    "formal": 1,
    # "humorous": 2,
    # "romantic": 3
}

# pre-load model with n styles
start_time = time.time()
log("loading model", level="debug")
model_fname_prefix = 'transformer_lm'
model_config_fpath = f"checkpoints/{model_fname_prefix}/config.json"
model_config = read_json(model_config_fpath)
model, tokenizer, train_data = initialize_inference_model(config=model_config)
log("model created and weights loaded", level="debug")
log(f"model initialization and loading took {time.time()-start_time} seconds", level="debug")

@app.route("/")
def homepage():
    return render_template("translation.html")

@app.route("/style-transfer", methods=["GET", "POST"])
def req_style_transfer():
    log("request received", level="debug")
    try:
        if request.is_json:
            data = request.get_json()
            styles = data["styles"]
            text = data["input_text"]
        else:
            styles = request.values.getlist("styles")
            text = request.values.get("input_text")
        log(f'styles: {styles}', level='debug')
        style_ids = []
        for style in styles:
            if style not in STYLES.keys():
                raise Exception(f"style {style} not supported")
            else:
                style_ids.append(STYLES[style])
        if not text:
            raise Exception("no input text found")
        log(f'style ids: {style_ids}', level='debug')

        log("predicting text", level="debug")
        start_time = time.time()
        pred_text = predict_text(text, 
            tokenizer, 
            style_ids,
            model_config,
            model,
            train_data=train_data,
        )
        log(f"prediction took {time.time()-start_time} seconds", level="debug")

        out = {"output_text": pred_text, "styles": styles}
        out.update(request.values)
        return jsonify(out)

    except Exception as err:
        log(err, level="error")
        log("request values:", dict(request.values.items()), level="error")
        return jsonify({"error": str(err), "request": request.values})


if __name__ == "__main__":
    log("starting flask app", level="debug")
    app.run(host="0.0.0.0", debug=True, threaded=False)
