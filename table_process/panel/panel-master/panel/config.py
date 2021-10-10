"""
The config module supplies the global config object and the extension
which provides convenient support for  loading and configuring panel
components.
"""
import ast
import copy
import inspect
import os
import sys

from contextlib import contextmanager
from weakref import WeakKeyDictionary

import param

from pyviz_comms import (
    JupyterCommManager as _JupyterCommManager, extension as _pyviz_extension
)

from .io.logging import panel_log_handler
from .io.notebook import load_notebook
from .io.state import state

__version__ = str(param.version.Version(
    fpath=__file__, archive_commit="$Format:%h$", reponame="panel"))

_LOCAL_DEV_VERSION = any(v in __version__ for v in ('post', 'dirty'))

#---------------------------------------------------------------------
# Public API
#---------------------------------------------------------------------

_PATH = os.path.abspath(os.path.dirname(__file__))


def validate_config(config, parameter, value):
    """
    Validates parameter setting on a hidden config parameter.
    """
    if config._validating:
        return
    config._validating = True
    orig = getattr(config, parameter)
    try:
        setattr(config, parameter, value)
    except Exception as e:
        raise e
    finally:
        setattr(config, parameter, orig)
        config._validating = False


class _base_config(param.Parameterized):

    css_files = param.List(default=[], doc="""
        External CSS files to load.""")

    js_files = param.Dict(default={}, doc="""
        External JS files to load. Dictionary should map from exported
        name to the URL of the JS file.""")

    js_modules = param.Dict(default={}, doc="""
        External JS fils to load as modules. Dictionary should map from
        exported name to the URL of the JS file.""")

    raw_css = param.List(default=[], doc="""
        List of raw CSS strings to add to load.""")


class _config(_base_config):
    """
    Holds global configuration options for Panel. The options can be
    set directly on the global config instance, via keyword arguments
    in the extension or via environment variables. For example to set
    the embed option the following approaches can be used:

        pn.config.embed = True

        pn.extension(embed=True)

        os.environ['PANEL_EMBED'] = 'True'
    """

    apply_signatures = param.Boolean(default=True, doc="""
        Whether to set custom Signature which allows tab-completion
        in some IDEs and environments.""")

    autoreload = param.Boolean(default=False, doc="""
        Whether to autoreload server when script changes.""")

    loading_spinner = param.Selector(default='arcs', objects=[
        'arc', 'arcs', 'bar', 'dots', 'petal'], doc="""
        Loading indicator to use when component loading parameter is set.""")

    loading_color = param.Color(default='#c3c3c3', doc="""
        Color of the loading indicator.""")

    profiler = param.Selector(default=None, allow_None=True, objects=[
        'pyinstrument', 'snakeviz'], doc="""
        The profiler engine to enable.""")

    safe_embed = param.Boolean(default=False, doc="""
        Ensure all bokeh property changes trigger events which are
        embedded. Useful when only partial updates are made in an
        app, e.g. when working with HoloViews.""")

    session_history = param.Integer(default=0, bounds=(-1, None), doc="""
        If set to a non-negative value this determines the maximum length
        of the pn.state.session_info dictionary, which tracks
        information about user sessions. A value of -1 indicates an
        unlimited history.""")

    sizing_mode = param.ObjectSelector(default=None, objects=[
        'fixed', 'stretch_width', 'stretch_height', 'stretch_both',
        'scale_width', 'scale_height', 'scale_both', None], doc="""
        Specify the default sizing mode behavior of panels.""")

    template = param.ObjectSelector(default=None, doc="""
        The default template to render served applications into.""")

    theme = param.ObjectSelector(default='default', objects=['default', 'dark'], doc="""
        The theme to apply to the selected global template.""")

    throttled = param.Boolean(default=False, doc="""
        If sliders and inputs should be throttled until release of mouse.""")

    _comms = param.ObjectSelector(
        default='default', objects=['default', 'ipywidgets', 'vscode', 'colab'], doc="""
        Whether to render output in Jupyter with the default Jupyter
        extension or use the jupyter_bokeh ipywidget model.""")

    _console_output = param.ObjectSelector(default='accumulate', allow_None=True,
                                 objects=['accumulate', 'replace', 'disable',
                                          False], doc="""
        How to log errors and stdout output triggered by callbacks
        from Javascript in the notebook.""")

    _cookie_secret = param.String(default=None, doc="""
        Configure to enable getting/setting secure cookies.""")

    _embed = param.Boolean(default=False, allow_None=True, doc="""
        Whether plot data will be embedded.""")

    _embed_json = param.Boolean(default=False, doc="""
        Whether to save embedded state to json files.""")

    _embed_json_prefix = param.String(default='', doc="""
        Prefix for randomly generated json directories.""")

    _embed_load_path = param.String(default=None, doc="""
        Where to load json files for embedded state.""")

    _embed_save_path = param.String(default='./', doc="""
        Where to save json files for embedded state.""")

    _log_level = param.Selector(
        default='WARNING', objects=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        doc="Log level of Panel loggers")

    _oauth_provider = param.ObjectSelector(
        default=None, allow_None=True, objects=[], doc="""
        Select between a list of authentification providers.""")

    _oauth_key = param.String(default=None, doc="""
        A client key to provide to the OAuth provider.""")

    _oauth_secret = param.String(default=None, doc="""
        A client secret to provide to the OAuth provider.""")

    _oauth_jwt_user = param.String(default=None, doc="""
        The key in the ID JWT token to consider the user.""")

    _oauth_redirect_uri = param.String(default=None, doc="""
        A redirect URI to provide to the OAuth provider.""")

    _oauth_encryption_key = param.ClassSelector(default=None, class_=bytes, doc="""
        A random string used to encode OAuth related user information.""")

    _oauth_extra_params = param.Dict(default={}, doc="""
        Additional parameters required for OAuth provider.""")

    _inline = param.Boolean(default=_LOCAL_DEV_VERSION, allow_None=True, doc="""
        Whether to inline JS and CSS resources. If disabled, resources
        are loaded from CDN if one is available.""")

    _admin = param.Boolean(default=False, doc="Whether the admin panel was enabled.")

    _truthy = ['True', 'true', '1', True, 1]

    _session_config = WeakKeyDictionary()

    def __init__(self, **params):
        super().__init__(**params)
        self._validating = False
        for p in self.param:
            if p.startswith('_'):
                setattr(self, p+'_', None)
        if self.log_level:
            panel_log_handler.setLevel(self.log_level)

    @contextmanager
    def set(self, **kwargs):
        values = [(k, v) for k, v in self.param.get_param_values() if k != 'name']
        overrides = [(k, getattr(self, k+'_')) for k in self.param if k.startswith('_')]
        for k, v in kwargs.items():
            setattr(self, k, v)
        try:
            yield
        finally:
            self.param.set_param(**dict(values))
            for k, v in overrides:
                setattr(self, k+'_', v)

    def __setattr__(self, attr, value):
        from .io.state import state
        if not getattr(self, 'initialized', False) or (attr.startswith('_') and attr.endswith('_')) or attr == '_validating':
            return super().__setattr__(attr, value)
        value = getattr(self, f'_{attr}_hook', lambda x: x)(value)
        if state.curdoc is not None:
            if attr in self.param:
                validate_config(self, attr, value)
            elif f'_{attr}' in self.param:
                validate_config(self, f'_{attr}', value)
            else:
                raise AttributeError(f'{attr!r} is not a valid config parameter.')
            if state.curdoc not in self._session_config:
                self._session_config[state.curdoc] = {}
            self._session_config[state.curdoc][attr] = value
        elif f'_{attr}' in self.param and hasattr(self, f'_{attr}_'):
            validate_config(self, f'_{attr}', value)
            super().__setattr__(f'_{attr}_', value)
        else:
            super().__setattr__(attr, value)

    @param.depends('_log_level', watch=True)
    def _update_log_level(self):
        panel_log_handler.setLevel(self._log_level)

    def __getattribute__(self, attr):
        from .io.state import state
        init = super().__getattribute__('initialized')
        if init and not attr.startswith('__'):
            params = super().__getattribute__('param')
        else:
            params = []
        session_config = super().__getattribute__('_session_config')
        if state.curdoc and state.curdoc not in session_config:
            session_config[state.curdoc] = {}
        if (attr in ('raw_css', 'css_files', 'js_files', 'js_modules') and
            state.curdoc and attr not in session_config[state.curdoc]):
            new_obj = copy.copy(super().__getattribute__(attr))
            setattr(self, attr, new_obj)
        if state.curdoc and state.curdoc in session_config and attr in session_config[state.curdoc]:
            return session_config[state.curdoc][attr]
        elif f'_{attr}' in params and getattr(self, f'_{attr}_') is not None:
            return super().__getattribute__(f'_{attr}_')
        return super().__getattribute__(attr)

    def _console_output_hook(self, value):
        return value if value else 'disable'

    def _template_hook(self, value):
        if isinstance(value, str):
            return self.param.template.names[value]
        return value

    @property
    def _doc_build(self):
        return os.environ.get('PANEL_DOC_BUILD')

    @property
    def console_output(self):
        if self._doc_build:
            return 'disable'
        else:
            return os.environ.get('PANEL_CONSOLE_OUTPUT', _config._console_output)

    @property
    def embed(self):
        return os.environ.get('PANEL_EMBED', _config._embed) in self._truthy

    @property
    def comms(self):
        return os.environ.get('PANEL_COMMS', _config._comms)

    @property
    def embed_json(self):
        return os.environ.get('PANEL_EMBED_JSON', _config._embed_json) in self._truthy

    @property
    def embed_json_prefix(self):
        return os.environ.get('PANEL_EMBED_JSON_PREFIX', _config._embed_json_prefix)

    @property
    def embed_save_path(self):
        return os.environ.get('PANEL_EMBED_SAVE_PATH', _config._embed_save_path)

    @property
    def embed_load_path(self):
        return os.environ.get('PANEL_EMBED_LOAD_PATH', _config._embed_load_path)

    @property
    def inline(self):
        return os.environ.get('PANEL_INLINE', _config._inline) in self._truthy

    @property
    def log_level(self):
        log_level = os.environ.get('PANEL_LOG_LEVEL', self._log_level)
        return log_level.upper() if log_level else None

    @property
    def oauth_provider(self):
        provider = os.environ.get('PANEL_OAUTH_PROVIDER', _config._oauth_provider)
        return provider.lower() if provider else None

    @property
    def oauth_key(self):
        return os.environ.get('PANEL_OAUTH_KEY', _config._oauth_key)

    @property
    def cookie_secret(self):
        return os.environ.get(
            'PANEL_COOKIE_SECRET',
            os.environ.get('BOKEH_COOKIE_SECRET', _config._cookie_secret)
        )

    @property
    def oauth_secret(self):
        return os.environ.get('PANEL_OAUTH_SECRET', _config._oauth_secret)

    @property
    def oauth_redirect_uri(self):
        return os.environ.get('PANEL_OAUTH_REDIRECT_URI', _config._oauth_redirect_uri)

    @property
    def oauth_jwt_user(self):
        return os.environ.get('PANEL_OAUTH_JWT_USER', _config._oauth_jwt_user)

    @property
    def oauth_encryption_key(self):
        return os.environ.get('PANEL_OAUTH_ENCRYPTION', _config._oauth_encryption_key)

    @property
    def oauth_extra_params(self):
        if 'PANEL_OAUTH_EXTRA_PARAMS' in os.environ:
            return ast.literal_eval(os.environ['PANEL_OAUTH_EXTRA_PARAMS'])
        else:
            return _config._oauth_extra_params


if hasattr(_config.param, 'objects'):
    _params = _config.param.objects()
else:
    _params = _config.param.params()

config = _config(**{k: None if p.allow_None else getattr(_config, k)
                    for k, p in _params.items() if k != 'name'})


class panel_extension(_pyviz_extension):
    """
    Initializes the pyviz notebook extension to allow plotting with
    bokeh and enable comms.
    """

    _loaded = False

    _imports = {
        'katex': 'panel.models.katex',
        'mathjax': 'panel.models.mathjax',
        'plotly': 'panel.models.plotly',
        'deckgl': 'panel.models.deckgl',
        'vega': 'panel.models.vega',
        'vtk': 'panel.models.vtk',
        'ace': 'panel.models.ace',
        'echarts': 'panel.models.echarts',
        'ipywidgets': 'ipywidgets_bokeh.widget',
        'perspective': 'panel.models.perspective',
        'terminal': 'panel.models.terminal',
        'tabulator': 'panel.models.tabulator',
        'gridstack': 'panel.layout.gridstack'
    }

    # Check whether these are loaded before rendering (if any item
    # in the list is available the extension will be confidered as
    # loaded)
    _globals = {
        'deckgl': ['deck'],
        'echarts': ['echarts'],
        'katex': ['katex'],
        'mathjax': ['MathJax'],
        'plotly': ['Plotly'],
        'vega': ['vega'],
        'vtk': ['vtk'],
        'terminal': ['Terminal', 'xtermjs'],
        'tabulator': ['Tabulator'],
        'gridstack': ['GridStack']
    }

    _loaded_extensions = []

    def __call__(self, *args, **params):
        # Abort if IPython not found
        for arg in args:
            if arg not in self._imports:
                self.param.warning('%s extension not recognized and '
                                   'will be skipped.' % arg)
            else:
                __import__(self._imports[arg])

        for k, v in params.items():
            if k in ['raw_css', 'css_files']:
                if not isinstance(v, list):
                    raise ValueError('%s should be supplied as a list, '
                                     'not as a %s type.' %
                                     (k, type(v).__name__))
                getattr(config, k).extend(v)
            elif k == 'js_files':
                getattr(config, k).update(v)
            else:
                setattr(config, k, v)

        if config.apply_signatures and sys.version_info.major >= 3:
            self._apply_signatures()

        loaded = self._loaded

        # Short circuit pyvista extension load if VTK is already initialized
        if loaded and args == ('vtk',) and 'vtk' in self._loaded_extensions:
            curframe = inspect.currentframe()
            calframe = inspect.getouterframes(curframe, 2)
            if len(calframe) >= 3 and 'pyvista' in calframe[2].filename:
                return

        if 'holoviews' in sys.modules:
            import holoviews as hv
            import holoviews.plotting.bokeh # noqa
            loaded = loaded or getattr(hv.extension, '_loaded', False)

            if hv.Store.current_backend in hv.Store.renderers:
                backend = hv.Store.current_backend
            else:
                backend = 'bokeh'
            if hasattr(hv.Store, 'set_current_backend'):
                hv.Store.set_current_backend(backend)
            else:
                hv.Store.current_backend = backend

        try:
            ip = params.pop('ip', None) or get_ipython() # noqa (get_ipython)
        except Exception:
            return

        newly_loaded = [arg for arg in args if arg not in panel_extension._loaded_extensions]
        if loaded and newly_loaded:
            self.param.warning(
                "A HoloViz extension was loaded previously. This means "
                "the extension is already initialized and the following "
                "Panel extensions could not be properly loaded: %s. "
                "If you are loading custom extensions with pn.extension(...) "
                "ensure that this is called before any other HoloViz "
                "extension such as hvPlot or HoloViews." % newly_loaded)
        else:
            panel_extension._loaded_extensions += newly_loaded

        if hasattr(ip, 'kernel') and not self._loaded and not config._doc_build:
            # TODO: JLab extension and pyviz_comms should be changed
            #       to allow multiple cleanup comms to be registered
            _JupyterCommManager.get_client_comm(self._process_comm_msg,
                                                "hv-extension-comm")
            state._comm_manager = _JupyterCommManager

        if 'ipywidgets' in sys.modules and config.embed:
            # In embedded mode the ipywidgets_bokeh model must be loaded
            __import__(self._imports['ipywidgets'])

        nb_load = False
        if 'holoviews' in sys.modules:
            if getattr(hv.extension, '_loaded', False):
                return
            with param.logging_level('ERROR'):
                hv.plotting.Renderer.load_nb(config.inline)
                if hasattr(hv.plotting.Renderer, '_render_with_panel'):
                    nb_load = True

        if not nb_load and hasattr(ip, 'kernel'):
            load_notebook(config.inline)
        panel_extension._loaded = True

        if 'comms' in params:
            return

        # Try to detect environment so that we can enable comms
        try:
            import google.colab # noqa
            config.comms = "colab"
            return
        except ImportError:
            pass

        # Check if we're running in VSCode
        if "VSCODE_PID" in os.environ:
            config.comms = "vscode"

    def _apply_signatures(self):
        from inspect import Parameter, Signature
        from .viewable import Viewable

        descendants = param.concrete_descendents(Viewable)
        for cls in reversed(list(descendants.values())):
            if cls.__doc__.startswith('params'):
                prefix = cls.__doc__.split('\n')[0]
                cls.__doc__ = cls.__doc__.replace(prefix, '')
            sig = inspect.signature(cls.__init__)
            sig_params = list(sig.parameters.values())
            if not sig_params or sig_params[-1] != Parameter('params', Parameter.VAR_KEYWORD):
                continue
            parameters = sig_params[:-1]

            processed_kws, keyword_groups = set(), []
            for cls in reversed(cls.mro()):
                keyword_group = []
                for (k, v) in sorted(cls.__dict__.items()):
                    if (isinstance(v, param.Parameter) and k not in processed_kws
                        and not v.readonly):
                        keyword_group.append(k)
                        processed_kws.add(k)
                keyword_groups.append(keyword_group)

            parameters += [
                Parameter(name, Parameter.KEYWORD_ONLY)
                for kws in reversed(keyword_groups) for name in kws
                if name not in sig.parameters
            ]
            kwarg_name = '_kwargs' if 'kwargs' in processed_kws else 'kwargs'
            parameters.append(Parameter(kwarg_name, Parameter.VAR_KEYWORD))
            cls.__init__.__signature__ = Signature(
                parameters, return_annotation=sig.return_annotation
            )


#---------------------------------------------------------------------
# Private API
#---------------------------------------------------------------------

def _cleanup_panel(msg_id):
    """
    A cleanup action which is called when a plot is deleted in the notebook
    """
    if msg_id not in state._views:
        return
    viewable, model, _, _ = state._views.pop(msg_id)
    viewable._cleanup(model)


def _cleanup_server(server_id):
    """
    A cleanup action which is called when a server is deleted in the notebook
    """
    if server_id not in state._servers:
        return
    server, viewable, docs = state._servers.pop(server_id)
    server.stop()
    for doc in docs:
        for root in doc.roots:
            if root.ref['id'] in viewable._models:
                viewable._cleanup(root)


panel_extension.add_delete_action(_cleanup_panel)
if hasattr(panel_extension, 'add_server_delete_action'):
    panel_extension.add_server_delete_action(_cleanup_server)
