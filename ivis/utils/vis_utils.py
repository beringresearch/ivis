import sys

try:
	import param
except ImportError:
	param = None

try:
	import panel as pn
except ImportError:
	pn = None

try:
	import holoviews as hv
	from holoviews.operation.datashader import datashade
	hv.extension('bokeh')
except ImportError:
	hv = None

try:
	import datashader as ds
except ImportError:
	ds = None

def check_dependencies():
	if hv is None:
		return False
	if ds is None:
		return False
	if param is None:
		return False
	if pn is None:
		return False
	return True


class EmbeddingsExplorer(param.Parameterized):
    label_flag = param.Boolean(default=True)

    def __init__(self):
        super(EmbeddingsExplorer, self).__init__()
        if not check_dependencies():
            message = (
                'Failed to import dependencies. You must `pip install holoviews datashader` '
                'for `EmbeddingsExplorer` to work.')
            raise ImportError(message)            

    @param.depends('label_flag')
    def get_points(self):
        embeddings = self.embeddings
        classes = self.classes
        if (self.label_flag) and (classes is not None):
            data = pd.DataFrame(embeddings)
            data.columns = ['ivis 1', 'ivis 2']
            data['label'] = classes
            num_ks = len(np.unique(classes))
            color_key = list(enumerate(Sets1to3[0:num_ks]))
            
            embed = {k: hv.Points(data.values[classes==k,:], ['ivis 1', 'ivis 2'], 'k',
                                  label=str(k)).opts(color=v, size=0) for k, v in color_key}
            dse = dynspread(datashade(hv.NdOverlay(embed, kdims=['k']), aggregator=ds.by('k', ds.count())))
            color_points = hv.NdOverlay({k: hv.Points([0,0]).opts(color=v, size=0) for k, v in color_key})
            points = color_points * dse
        else:
            points = datashade(hv.Points(embeddings))
        
        points.opts(height=400, width=500, xaxis=None, yaxis=None)
        return points
    
    def view(self, embeddings, classes=None):
        self.embeddings = embeddings
        self.classes = classes
        return pn.Row(self.get_points, self.param)