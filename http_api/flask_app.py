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
app.json_encoder = DataEncoder
app.logger.debug('debug')

# mapping from style names to style transfer model objects
# each model object contains: 1) weights, 2) config, 3) source corpus, 4) target corpus
MODELS = {}

# mapping from style names to prefix strings (used in fnames)
STYLE_DICT_FORWARD = {}
STYLE_DICT_BACKWARD = {}


def get_model(style_name):
    if style_name in MODELS:
        log("returning preloaded model", level="debug")
        return MODELS[style_name]

    log("loading new model", level="debug")
    if style_name in STYLE_DICT_FORWARD:
        model_fname_prefix = STYLE_DICT_FORWARD[style_name]
    else: 
        model_fname_prefix = STYLE_DICT_BACKWARD[style_name]
    model_config_fpath = f"checkpoints/{model_fname_prefix}/config.json"
    model_config = read_json(model_config_fpath)

    start_time = time.time()
    del_and_ret_model, src, tgt = initialize_inference_model(config=model_config)
    log("model created", level="debug")
    del_and_ret_model, _ = attempt_load_model(
        model=del_and_ret_model,
        checkpoint_dir=f"checkpoints/{model_fname_prefix}",
        map_location=torch.device('cpu')
    )
    log("model weights loaded", level="debug")
    log(f"{style_name} model initialization and loading took {time.time()-start_time} seconds", level="debug")

    # store the model information in memory before returning
    MODELS[style_name] = {}
    MODELS[style_name]['model'] = del_and_ret_model
    MODELS[style_name]['config'] = model_config
    MODELS[style_name]['src'] = src
    MODELS[style_name]['tgt'] = tgt

    return MODELS[style_name]


#pre-load formal and romantic models
STYLE_DICT_FORWARD["formal"] = "del_and_ret-formal"
STYLE_DICT_BACKWARD["informal"] = "del_and_ret-formal"
log("loading formal and informal models", level="debug")
MODELS["informal"] = get_model("formal")
assert "formal" in MODELS
assert "informal" in MODELS

STYLE_DICT_FORWARD["romantic"] = "del_and_ret-romantic"
STYLE_DICT_BACKWARD["humorous"] = "del_and_ret-formal"
log("loading romantic and humorous models", level="debug")
MODELS["humorous"] = get_model("romantic")
assert "romantic" in MODELS
assert "humorous" in MODELS


@app.route("/")
def homepage():
    with open("http_api/readme.md") as readme:
        content = readme.read()

    content = Markup(markdown.markdown(content))
    return render_template("templates/index.pug", content=content)


@app.route("/style-transfer", methods=["GET", "POST"])
def req_style_transfer(read_test_data=False):
    log("request received", level="debug")
    try:
        style = request.values.get("style", "formal")
        if style not in STYLE_DICT_FORWARD and style not in STYLE_DICT_BACKWARD:
            raise Exception(f"style {style} not supported")

        text = request.values.get("input_text")
        if not text:
            raise Exception("no input text found")

        start_time = time.time()
        log("retreiving model and loading style corpus'", level="debug")
        del_and_ret_model = get_model(style)

        log("predicting text", level="debug")
        if style in STYLE_DICT_FORWARD:
            forward = True
            tgt = del_and_ret_model['tgt']
            src = del_and_ret_model['src']
        else:
            forward = False
            tgt = del_and_ret_model['src']
            src = del_and_ret_model['tgt']
        pred_text = predict_text(text, 
            del_and_ret_model['model'], 
            src,
            tgt,
            del_and_ret_model['config'],
            forward=forward
        )
        log(f"prediction took {time.time()-start_time} seconds", level="debug")

        out = {"output_text": pred_text, "style": style}
        out.update(request.values)
        return jsonify(out)

    except Exception as err:
        log(err, level="error")
        log("request values:", dict(request.values.items()), level="error")
        return jsonify({"error": str(err), "request": request.values})


if __name__ == "__main__":
    log("starting flask app", level="debug")
    app.run(host="0.0.0.0", debug=True, threaded=False)
