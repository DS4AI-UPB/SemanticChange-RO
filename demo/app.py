from flask import Flask, render_template, request
from os import listdir
import json
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import json
import csv
import numpy as np
from sklearn.manifold import TSNE

app = Flask(__name__)

def get_represenations():
    results = []
    representations = listdir('data')
    for result_directory in representations:
        disp_name = result_directory.replace('_', ' ')
        results.append({'name': disp_name, 'value': result_directory})
    model = {"repressentations": results}
    return model

def get_embeddings(file_name, k = 40):
    try:
        keys = []
        embeddings = []

        with open(file_name, 'r') as f:
            csv_reader = csv.reader(f, delimiter=',')
            sup = 0
            for row in csv_reader:
                keys.append(row[0])
                embeddings.append([row[1:]])
                sup += 1
                if sup == k:
                    break
        tsne_model_3d = TSNE(n_components=3, learning_rate='auto', init='random')
        embeddings = np.array(embeddings)
        n, m, k = embeddings.shape
        embeddings_3d = np.array(tsne_model_3d.fit_transform(embeddings.reshape(n * m, k))).reshape(n, m, 3)
        embeddings_3d = [embedding[0] for embedding in embeddings_3d]
        return embeddings_3d, keys
    except:
        return None, None

@app.route('/compare')
def redirect():
    model = {}
    graphJSON = None
    if len(request.args) > 0:
        names = []
        tps = []
        fps = []
        tns = []
        fns = []
        for arg in request.args:
            data = json.load(open(f"data/{arg}/desc.json", 'r'))
            for results in data['results']:
                names.append(f"{data['name']}_{results['metric']}")
                tps.append(results['true_positives'])
                fps.append(results['false_positives'])
                tns.append(results['true_negatives'])
                fns.append(results['false_negatives'])
        go_data = []
        for name in names:
            go_data.append(go.Bar( x=['true positives', 'false positives', 'true negatives', 'false negatives'],
                                y=[tps[names.index(name)], fps[names.index(name)], tns[names.index(name)], fns[names.index(name)]],
                            name = name))
        fig = go.Figure(data=go_data)        
        fig.update_layout(barmode='group', xaxis_tickangle=-45)
        fig.update_xaxes(title_text="Metric")
        fig.update_yaxes(title_text="Count")
        fig.update_layout(height=800, width=800)
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    else:
        model = get_represenations()
        model["no_models"] = True
    model.update(get_represenations())
    return render_template('compare.html', graphJSON=graphJSON, model=model)

@app.route('/compare_select')
def compare_select():
    return render_template('compare_select.html', model=get_represenations())

@app.route('/results')
def results():
    args = request.args
    model = {}
    graphJSON = None
    if 'results' in args:
        result_directory = args['results']
        model = json.load(open(f"data/{result_directory}/desc.json", 'r'))
        model["root"] = f"data/{result_directory}/"
        embeddings, keys = get_embeddings(f"data/{result_directory}/words.csv")
        if embeddings is not None:
            keys = [key[:-2] for key in keys]

            key2embs = {}
            emb2key = {}
            for key, embedding in zip(keys, embeddings):
                if key not in key2embs:
                    key2embs[key] = []
                key2embs[key].append(embedding)
                emb2key[str(embedding)] = key
            key2diff = {}
            for key in keys:
                key2diff[key] = np.linalg.norm(np.array(key2embs[key][1]) - np.array(key2embs[key][0]))

            diffs = []
            for embedding in embeddings:
                diffs.append(key2diff[emb2key[str(embedding)]])
            print(diffs)
            
            e1s = [embedding[0] for embedding in embeddings]
            e2s = [embedding[1] for embedding in embeddings]
            e3s = [embedding[2] for embedding in embeddings]
            embs = [[e1,e2,e3,w] for e1,e2,e3,w in zip(e1s, e2s, e3s, keys)]
            df = pd.DataFrame(embs, columns=['x', 'y', 'z', 'word'])
            fig = px.line_3d(df, x='x', y='y', z='z', color='word', width=1200, height=1200, hover_name= diffs)
            graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        else:
            model.update({"no_embeddings": True})
        model.update(get_represenations())
        return render_template('results.html',graphJSON=graphJSON, model = model)
    else:
        return render_template('results.html', model = {})

@app.route('/')
def index():
    return render_template('index.html', model = get_represenations())

if __name__ == '__main__':
    app.run(debug=True)