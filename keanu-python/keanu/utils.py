import pandas as pd
from keanu.base import Vertex

def samples_to_dataframe(net, samples, model=None):
    latentVertexIds = [v.getId() for v in net.getLatentVertices()]
    results = dict([(vertexId, [e for e in samples.get(vertexId).asList()]) for vertexId in latentVertexIds])

    if model:
        mapping = {}
        for k, v in model._vertices.items():
            if isinstance(v, Vertex):
                mapping[v.getId()] = k

        remapped_results = {}
        for k, v in results.items():
            if k in mapping:
                remapped_results[mapping[k]] = v

        results = remapped_results

    return pd.DataFrame.from_dict(results)
