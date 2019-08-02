# Delete and Retrieve Style Transfer

Run `make version=<version> api` to start the app if running it from the github repo. The `delete_and_retrieve-api` image should start the flask app on startup by default. 

## Routes

To rewrite text in a given style, make a GET or POST request to `/style-transfer`. The parameters `input_text` and `style` can be specified either in the url query string of a GET request or in the body of a POST request. If `style` is not specified, it will default to `formal` (currently the only style supported). 

    localhost:5000/style-transfer?input_text=this ain't no sentence&style=formal
    
The value returned should be a json object containing the resulting `output_text` as well as the parameters used in the request:

    {"input_text":"this ain't no sentence", "output_text": "this is not a sentence", "style": "formal"}    
    
