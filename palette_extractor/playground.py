# from paletteSampler import PaletteSampler

import os
import webview
import threading
import base64
from paletteSampler import PaletteSampler


html = """
<!DOCTYPE html>
<html>
<head lang="en">
<meta charset="UTF-8">
<style>
    #response-container {
        display: none;
        padding: 1rem;
        margin: 1rem 2rem;
        font-size: 120%;
        border: 5px dashed #ccc;
    }
    #response-image {
        height: auto;
        width: auto;
        max-width: 90vw;
        max-height: 80vh;
        padding: 5px;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    h1 {
        text-align: center;
    }
    #main-div{
        text-align: center;
    }
    button {
        font-size: 100%;
        padding: 0.5rem;
        margin: 0.3rem;
        text-transform: uppercase;
    }
</style>
</head>
<body>
<h1>Pallet</h1>
<div id="main-div">
<button onClick="openFile()">Open file dialog</button>
<label for="steps">Numero de cores: </label><input id="steps" type="number" min="3" max="15" step="1" value ="5"/>
<label for="percent">Largura da paleta %: </label><input id="percent" type="number" min="10" max="100" step="5" value ="20"/>
<button onClick="accept()">Accept</button><br/>
</div> 
<div id="response-container"></div>
<img id="response-image">
<script>
function showResponse(response) {
    var container = document.getElementById('response-container')
    container.innerText = response.message
    container.style.display = 'block'
}
function showImage(response) {
    var image = document.getElementById('response-image')
    image.src = response.image
}
function accept() {
    var selector = document.getElementById('steps')
    var percent = document.getElementById('percent')
    var params = {"selector": selector.value, "percent": percent.value}
    pywebview.api.accept(params).then(showResponse)
    run()
}
function run() {
    pywebview.api.run().then(showImage)
}
function openFile() {
    pywebview.api.open_file_dialog().then(showImage)
}
</script>
</body>
</html>
"""

class Api:
    def __init__(self):
        self._n = 5
        self._new_file = None
        self._percent = 0.1

    def setNumberOfPalletes(self, params):
        response = {
            'message': 'Numero de cores selecionadas {n}'.format(n=self._n)
        }
        return response

    def setPercent(self, params):
        response = {
            'message': 'Numero de cores selecionadas {n}'.format(n=self._percent)
        }
        return response

    def accept(self, params):
        self._n = int(params["selector"])
        self._percent = float(params["percent"]) / 100
        print(self._n, self._percent, params)
        response = {
            'message': 'Processando...'
        }
        return response

    def run(self, params):
        image_path = self._new_file
        n = self._n
        percent = self._percent
        app = PaletteSampler()
        app.load_image(image_path)
        output_path = os.path.join(os.path.split(image_path)[0], "sampled_" + str(n) + "_" + os.path.split(image_path)[1])
        app.render(n=n, percent=percent, output=output_path)

        response = {
            'message': 'File saved - {file}'.format(file=output_path),
            'image': output_path,
        }
        return response

    def open_file_dialog(self, params):
        file_types = ('Image Files (*.bmp;*.jpg;*.gif)', 'All files (*.*)')

        self._new_file = webview.create_file_dialog(webview.OPEN_DIALOG,
                                         allow_multiple=True,
                                         file_types=file_types)[0]

        response = {
                'message': 'Loaded file {file}'.format(file=self._new_file),
                'image': self._new_file
            }
        return response

def create_app():
    webview.load_html(html)


if __name__ == '__main__':
    t = threading.Thread(target=create_app)
    t.start()

    api = Api()
    webview.create_window('API example', js_api=api, width=1200, height=600, resizable = True, min_size = (400, 200))