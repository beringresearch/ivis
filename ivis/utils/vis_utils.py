import pandas as pd
import numpy as np

try:
    import param
except ImportError:
    raise ImportError(
        'Failed to import param. You must run '
        '`pip install param` to use vis_utils')

try:
    import panel as pn
except ImportError:
    pn = None

try:
    import holoviews as hv
    from holoviews.operation.datashader import datashade, dynspread
    hv.extension('bokeh')
except ImportError:
    hv = None

try:
    import datashader as ds
    from datashader.colors import Sets1to3
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
    """Interactive visualisation of embeddings.

    Displays a scatterplot that can be embedded inside a Jupyter notebook.
    """

    label_flag = param.Boolean(default=True)

    def __init__(self):
        super(EmbeddingsExplorer, self).__init__()
        if not check_dependencies():
            message = (
                'Failed to import dependencies. You must '
                '`pip install holoviews datashader bokeh` '
                'for `EmbeddingsExplorer` to work.')
            raise ImportError(message)
        self.embeddings = None
        self.classes = None

    @param.depends('label_flag')
    def _get_points(self):
        embeddings = self.embeddings
        classes = self.classes
        if (self.label_flag) and (classes is not None):
            data = pd.DataFrame(embeddings)
            data.columns = ['ivis 1', 'ivis 2']
            data['label'] = classes
            num_ks = len(np.unique(classes))
            color_key = list(enumerate(Sets1to3[0:num_ks]))

            embed = {k: hv.Points(data.values[classes == k, :], ['ivis 1', 'ivis 2'], 'k',
                                  label=str(k)).opts(color=v, size=0) for k, v in color_key}
            dse = dynspread(
                datashade(hv.NdOverlay(embed, kdims=['k']), aggregator=ds.by('k', ds.count())))
            color_points = hv.NdOverlay(
                {k: hv.Points([0, 0]).opts(color=v, size=0) for k, v in color_key})
            points = color_points * dse
        else:
            points = datashade(hv.Points(embeddings))

        points.opts(height=400, width=500, xaxis=None, yaxis=None)
        return points

    def view(self, embeddings, classes=None):
        """Visualise embeddings

        Parameters
        ----------
        embeddings : n-dimensional array shape (n_samples, )
            Data to be visualised.
        classes : array, shape (n_samples)
            Optional array of class labels to be overlayed on the scatterplot.

        Returns
        -------
        An instance of panel.Row
        """
        self.embeddings = embeddings
        self.classes = classes
        return pn.Row(self._get_points, self.param)
