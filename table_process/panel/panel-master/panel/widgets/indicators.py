import os
import sys
import warnings

from math import pi

import numpy as np
import param

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from tqdm.asyncio import tqdm as _tqdm

from ..layout import Column, Row
from ..models import (
    HTML, Progress as _BkProgress, TrendIndicator as _BkTrendIndicator
)
from ..pane.markup import Str
from ..reactive import SyncableData
from ..util import escape, updating
from ..viewable import Viewable
from .base import Widget

RED   = "#d9534f"
GREEN = "#5cb85c"
BLUE  = "#428bca"

class Indicator(Widget):
    """
    Indicator is a baseclass for widgets which indicate some state.
    """

    sizing_mode = param.ObjectSelector(default='fixed', objects=[
        'fixed', 'stretch_width', 'stretch_height', 'stretch_both',
        'scale_width', 'scale_height', 'scale_both', None])

    __abstract = True


class BooleanIndicator(Indicator):

    value = param.Boolean(default=False, doc="""
        Whether the indicator is active or not.""")

    __abstract = True


class BooleanStatus(BooleanIndicator):

    color = param.ObjectSelector(default='dark', objects=[
        'primary', 'secondary', 'success', 'info', 'danger', 'warning',
        'light', 'dark'])

    height = param.Integer(default=20, doc="""
        height of the circle.""")

    width = param.Integer(default=20, doc="""
        Width of the circle.""")

    value = param.Boolean(default=False, doc="""
        Whether the indicator is active or not.""")

    _rename = {'color': None}

    _source_transforms = {'value': None}

    _widget_type = HTML

    def _process_param_change(self, msg):
        msg = super()._process_param_change(msg)
        value = msg.pop('value', None)
        if value is None:
            return msg
        msg['css_classes'] = ['dot-filled', self.color] if value else ['dot']
        return msg


class LoadingSpinner(BooleanIndicator):

    bgcolor = param.ObjectSelector(default='light', objects=['dark', 'light'])

    color = param.ObjectSelector(default='dark', objects=[
        'primary', 'secondary', 'success', 'info', 'danger', 'warning',
        'light', 'dark'])

    height = param.Integer(default=125, doc="""
        height of the circle.""")

    width = param.Integer(default=125, doc="""
        Width of the circle.""")

    value = param.Boolean(default=False, doc="""
        Whether the indicator is active or not.""")

    _rename = {'color': None, 'bgcolor': None}

    _source_transforms = {'value': None}

    _widget_type = HTML

    def _process_param_change(self, msg):
        msg = super()._process_param_change(msg)
        value = msg.pop('value', None)
        if value is None:
            return msg
        color_cls = f'{self.color}-{self.bgcolor}'
        msg['css_classes'] = ['loader', 'spin', color_cls] if value else ['loader', self.bgcolor]
        return msg


class ValueIndicator(Indicator):
    """
    A ValueIndicator provides a visual representation for a numeric
    value.
    """

    value = param.Number(default=None, allow_None=True)

    __abstract = True


class Progress(ValueIndicator):

    active = param.Boolean(default=True, doc="""
        If no value is set the active property toggles animation of the
        progress bar on and off.""")

    bar_color = param.ObjectSelector(default='success', objects=[
        'primary', 'secondary', 'success', 'info', 'danger', 'warning',
        'light', 'dark'])

    max = param.Integer(default=100, doc="The maximum value of the progress bar.")

    value = param.Integer(default=-1, bounds=(-1, None),                           
                          allow_None = True,#TODO: remove
                          doc="""
        The current value of the progress bar. If set to -1 the progress
        bar will be indeterminate and animate depending on the active
        parameter.""")

    _rename = {'name': None}

    _widget_type = _BkProgress

    @param.depends('max', watch=True)
    def _update_value_bounds(self):
        self.param.value.bounds = (-1, self.max)
        
    @param.depends('value', watch=True)
    def _warn_deprecation(self):
        if self.value is None:
            self.value = -1
            warnings.warn('Setting the progress value to None is deprecated, use -1 instead.', 
                          FutureWarning,  stacklevel=2)  

    def __init__(self,**params):
        super().__init__(**params)
        self._update_value_bounds()


class Number(ValueIndicator):
    """
    The Number indicator renders the value as text optionally colored
    according to the color thresholds.
    """

    default_color = param.String(default='black')

    colors = param.List(default=None)

    format = param.String(default='{value}')

    font_size = param.String(default='54pt')

    nan_format = param.String(default='-', doc="""
      How to format nan values.""")

    title_size = param.String(default='18pt')

    _rename = {}

    _source_transforms = {
        'value': None, 'colors': None, 'default_color': None,
        'font_size': None, 'format': None, 'nan_format': None,
        'title_size': None
    }

    _widget_type = HTML

    def _process_param_change(self, msg):
        msg = super()._process_param_change(msg)
        font_size = msg.pop('font_size', self.font_size)
        title_font_size = msg.pop('title_size', self.title_size)
        name = msg.pop('name', self.name)
        format = msg.pop('format', self.format)
        value = msg.pop('value', self.value)
        nan_format = msg.pop('nan_format', self.nan_format)
        color = msg.pop('default_color', self.default_color)
        colors = msg.pop('colors', self.colors)
        for val, clr in (colors or [])[::-1]:
            if value is not None and value <= val:
                color = clr
        if value is None:
            value = float('nan')
        value = format.format(value=value).replace('nan', nan_format)
        text = f'<div style="font-size: {font_size}; color: {color}">{value}</div>'
        if self.name:
            title_font_size = msg.pop('title_size', self.title_size)
            text = f'<div style="font-size: {title_font_size}; color: {color}">{name}</div>\n{text}'
        msg['text'] = escape(text)
        return msg


class String(ValueIndicator):
    """
    The String indicator renders a string with a title.
    """

    default_color = param.String(default='black')

    font_size = param.String(default='54pt')

    title_size = param.String(default='18pt')

    value = param.String(default=None, allow_None=True)

    _rename = {}

    _source_transforms = {
        'value': None, 'default_color': None, 'font_size': None, 'title_size': None
    }

    _widget_type = HTML

    def _process_param_change(self, msg):
        msg = super()._process_param_change(msg)
        font_size = msg.pop('font_size', self.font_size)
        title_font_size = msg.pop('title_size', self.title_size)
        name = msg.pop('name', self.name)
        value = msg.pop('value', self.value)
        color = msg.pop('default_color', self.default_color)
        text = f'<div style="font-size: {font_size}; color: {color}">{value}</div>'
        if self.name:
            title_font_size = msg.pop('title_size', self.title_size)
            text = f'<div style="font-size: {title_font_size}; color: {color}">{name}</div>\n{text}'
        msg['text'] = escape(text)
        return msg


class Gauge(ValueIndicator):
    """
    A Gauge represents a value in some range as a position on
    speedometer or gauge. It is similar to a Dial but visually a lot
    busier.
    """

    annulus_width = param.Integer(default=10, doc="""
      Width of the gauge annulus.""")

    bounds = param.Range(default=(0, 100), doc="""
      The upper and lower bound of the dial.""")

    colors = param.List(default=None, doc="""
      Color thresholds for the Gauge, specified as a list of tuples
      of the fractional threshold and the color to switch to.""")

    custom_opts = param.Dict(doc="""
      Additional options to pass to the ECharts Gauge definition.""")

    height = param.Integer(default=300, bounds=(0, None))

    end_angle = param.Number(default=-45, doc="""
      Angle at which the gauge ends.""")

    format = param.String(default='{value}%', doc="""
      Formatting string for the value indicator.""")

    num_splits = param.Integer(default=10, doc="""
      Number of splits along the gauge.""")

    show_ticks = param.Boolean(default=True, doc="""
      Whether to show ticks along the dials.""")

    show_labels = param.Boolean(default=True, doc="""
      Whether to show tick labels along the dials.""")

    start_angle = param.Number(default=225, doc="""
      Angle at which the gauge starts.""")

    tooltip_format = param.String(default='{b} : {c}%', doc="""
      Formatting string for the hover tooltip.""")

    title_size = param.Integer(default=18, doc="""
      Size of title font.""")

    value = param.Number(default=25, doc="""
      Value to indicate on the gauge a value within the declared bounds.""")

    width = param.Integer(default=300, bounds=(0, None))

    _rename = {}

    _source_transforms = {
        'annulus_width': None, 'bounds': None, 'colors': None,
        'custom_opts': None, 'end_angle': None, 'format': None,
        'num_splits': None, 'show_ticks': None, 'show_labels': None,
        'start_angle': None, 'tooltip_format': None, 'title_size': None,
        'value': None
    }

    @property
    def _widget_type(self):
        if 'panel.models.echarts' not in sys.modules:
            from ..models.echarts import ECharts
        else:
            ECharts = getattr(sys.modules['panel.models.echarts'], 'ECharts')
        return ECharts

    def __init__(self, **params):
        super().__init__(**params)
        self._update_value_bounds()

    @param.depends('bounds', watch=True)
    def _update_value_bounds(self):
        self.param.value.bounds = self.bounds

    def _process_param_change(self, msg):
        msg = super()._process_param_change(msg)
        vmin, vmax = msg.pop('bounds', self.bounds)
        msg['data'] = {
            'tooltip': {
                'formatter': msg.pop('tooltip_format', self.tooltip_format)
            },
            'series': [{
                'name': 'Gauge',
                'type': 'gauge',
                'axisTick': {'show': msg.pop('show_ticks', self.show_ticks)},
                'axisLabel': {'show': msg.pop('show_labels', self.show_labels)},
                'title': {'fontWeight': 'bold', 'fontSize': msg.pop('title_size', self.title_size)},
                'splitLine': {'show': True},
                'radius': '100%',
                'detail': {'formatter': msg.pop('format', self.format)},
                'min': vmin,
                'max': vmax,
                'startAngle': msg.pop('start_angle', self.start_angle),
                'endAngle': msg.pop('end_angle', self.end_angle),
                'splitNumber': msg.pop('num_splits', self.num_splits),
                'data': [{'value': msg.pop('value', self.value), 'name': self.name}],
                'axisLine': {
                    'lineStyle': {
                        'width': msg.pop('annulus_width', self.annulus_width),
                    }
                }
            }]
        }
        colors = msg.pop('colors', self.colors)
        if colors:
            msg['data']['series'][0]['axisLine']['lineStyle']['color'] = colors
        custom_opts = msg.pop('custom_opts', self.custom_opts)
        if custom_opts:
            gauge = msg['data']['series'][0]
            for k, v in custom_opts.items():
                if k not in gauge or not isinstance(gauge[k], dict):
                    gauge[k] = v
                else:
                    gauge[k].update(v)
        return msg


class Dial(ValueIndicator):
    """
    A Dial represents a value in some range as a position on an
    annular dial. It is similar to a Gauge but more minimal visually.
    """

    annulus_width = param.Number(default=0.2, doc="""
      Width of the radial annulus as a fraction of the total.""")

    bounds = param.Range(default=(0, 100), doc="""
      The upper and lower bound of the dial.""")

    colors = param.List(default=None, doc="""
      Color thresholds for the Dial, specified as a list of tuples
      of the fractional threshold and the color to switch to.""")

    default_color = param.String(default='lightblue', doc="""
      Color of the radial annulus if not color thresholds are supplied.""")

    end_angle = param.Number(default=25, doc="""
      Angle at which the dial ends.""")

    format = param.String(default='{value}%', doc="""
      Formatting string for the value indicator and lower/upper bounds.""")

    height = param.Integer(default=250, bounds=(1, None))

    nan_format = param.String(default='-', doc="""
      How to format nan values.""")

    needle_color = param.String(default='black', doc="""
      Color of the Dial needle.""")

    needle_width = param.Number(default=0.1, doc="""
      Radial width of the needle.""")

    start_angle = param.Number(default=-205, doc="""
      Angle at which the dial starts.""")

    tick_size = param.String(default=None, doc="""
      Font size of the Dial min/max labels.""")

    title_size = param.String(default=None, doc="""
      Font size of the Dial title.""")

    unfilled_color = param.String(default='whitesmoke', doc="""
      Color of the unfilled region of the Dial.""")

    value_size = param.String(default=None, doc="""
      Font size of the Dial value label.""")

    value = param.Number(default=25, allow_None=True, doc="""
      Value to indicate on the dial a value within the declared bounds.""")

    width = param.Integer(default=250, bounds=(1, None))

    _manual_params = [
        'value', 'start_angle', 'end_angle', 'bounds',
        'annulus_width', 'format', 'background', 'needle_width',
        'tick_size', 'title_size', 'value_size', 'colors',
        'default_color', 'unfilled_color', 'height',
        'width', 'nan_format', 'needle_color'
    ]

    _data_params = _manual_params

    _rename = {'background': 'background_fill_color'}

    def __init__(self, **params):
        super().__init__(**params)
        self._update_value_bounds()

    @param.depends('bounds', watch=True)
    def _update_value_bounds(self):
        self.param.value.bounds = self.bounds

    def _get_data(self):
        vmin, vmax = self.bounds
        value = self.value
        if value is None:
            value = float('nan')
        fraction = (value-vmin)/(vmax-vmin)
        start = (np.radians(360-self.start_angle) - pi % (2*pi)) + pi
        end = (np.radians(360-self.end_angle) - pi % (2*pi)) + pi
        distance = (abs(end-start) % (pi*2))
        if end>start:
            distance = (pi*2)-distance
        radial_fraction = distance*fraction
        angle = start if np.isnan(fraction) else (start-radial_fraction)
        inner_radius = 1-self.annulus_width

        color = self.default_color
        for val, clr in (self.colors or [])[::-1]:
            if fraction <= val:
                color = clr

        annulus_data = {
            'starts': np.array([start, angle]),
            'ends' :  np.array([angle, end]),
            'color':  [color, self.unfilled_color],
            'radius': np.array([inner_radius, inner_radius])
        }

        x0s, y0s, x1s, y1s, clrs = [], [], [], [], []
        colors = self.colors or []
        for (val, _), (_, clr) in zip(colors[:-1], colors[1:]):
            tangle = start-(distance*val)
            if (vmin + val * (vmax-vmin)) <= value:
                continue
            x0, y0 = np.cos(tangle), np.sin(tangle)
            x1, y1 = x0*inner_radius, y0*inner_radius
            x0s.append(x0)
            y0s.append(y0)
            x1s.append(x1)
            y1s.append(y1)
            clrs.append(clr)

        threshold_data = {
            'x0': x0s, 'y0': y0s, 'x1': x1s, 'y1': y1s, 'color': clrs
        }

        center_radius = 1-self.annulus_width/2.
        x, y = np.cos(angle)*center_radius, np.sin(angle)*center_radius
        needle_start = pi+angle-(self.needle_width/2.)
        needle_end = pi+angle+(self.needle_width/2.)
        needle_data = {
            'x':      np.array([x]),
            'y':      np.array([y]),
            'start':  np.array([needle_start]),
            'end':    np.array([needle_end]),
            'radius': np.array([center_radius])
        }

        value = self.format.format(value=value).replace('nan', self.nan_format)
        min_value = self.format.format(value=vmin)
        max_value = self.format.format(value=vmax)
        tminx, tminy = np.cos(start)*center_radius, np.sin(start)*center_radius
        tmaxx, tmaxy = np.cos(end)*center_radius, np.sin(end)*center_radius
        tmin_angle, tmax_angle = start+pi, end+pi % pi
        scale = (self.height/400)
        title_size = self.title_size if self.title_size else '%spt' % (scale*32)
        value_size = self.value_size if self.value_size else '%spt' % (scale*48)
        tick_size = self.tick_size if self.tick_size else '%spt' % (scale*18)

        text_data= {
            'x':    np.array([0, 0, tminx, tmaxx]),
            'y':    np.array([-.2, -.5, tminy, tmaxy]),
            'text': [self.name, value, min_value, max_value],
            'rot':  np.array([0, 0, tmin_angle, tmax_angle]),
            'size': [title_size, value_size, tick_size, tick_size],
            'color': ['black', color, 'black', 'black']
        }
        return annulus_data, needle_data, threshold_data, text_data

    def _get_model(self, doc, root=None, parent=None, comm=None):
        params = self._process_param_change(self._init_params())
        model = figure(
            x_range=(-1,1), y_range=(-1,1), tools=[],
            outline_line_color=None, toolbar_location=None,
            width=self.width, height=self.height, **params
        )
        model.xaxis.visible = False
        model.yaxis.visible = False
        model.grid.visible = False

        annulus, needle, threshold, text = self._get_data()

        # Draw annulus
        annulus_source = ColumnDataSource(data=annulus, name='annulus_source')
        model.annular_wedge(
            x=0, y=0, inner_radius='radius', outer_radius=1, start_angle='starts',
            end_angle='ends', line_color='gray', color='color', direction='clock',
            source=annulus_source
        )

        # Draw needle
        needle_source = ColumnDataSource(data=needle, name='needle_source')
        model.wedge(
            x='x', y='y', radius='radius', start_angle='start', end_angle='end',
            fill_color=self.needle_color, line_color=self.needle_color,
            source=needle_source, name='needle_renderer'
        )

        # Draw thresholds
        threshold_source = ColumnDataSource(data=threshold, name='threshold_source')
        model.segment(
            x0='x0', x1='x1', y0='y0', y1='y1', line_color='color', source=threshold_source,
            line_width=2
        )

        # Draw labels
        text_source = ColumnDataSource(data=text, name='label_source')
        model.text(
            x='x', y='y', text='text', font_size='size', text_align='center',
            text_color='color', source=text_source, text_baseline='top',
            angle='rot'
        )

        if root is None:
            root = model
        self._models[root.ref['id']] = (model, parent)
        return model

    def _manual_update(self, events, model, doc, root, parent, comm):
        update_data = False
        for event in events:
            if event.name in ('width', 'height'):
                model.update(**{event.name: event.new})
            if event.name in self._data_params:
                update_data = True
            elif event.name == 'needle_color':
                needle_r = model.select(name='needle_renderer')
                needle_r.glyph.line_color = event.new
                needle_r.glyph.fill_color = event.new
        if not update_data:
            return
        annulus, needle, threshold, labels = self._get_data()
        model.select(name='annulus_source').data.update(annulus)
        model.select(name='needle_source').data.update(needle)
        model.select(name='threshold_source').data.update(threshold)
        model.select(name='label_source').data.update(labels)


class Trend(SyncableData, Indicator):
    """
    The Trend indicator enables the user to display a Dashboard KPI Card.

    The card can be layout out as:

    * a column (text and plot on top of each other) or
    * a row (text and plot after each other)

    The text section is responsive and resizes on window resize.
    """

    data = param.Parameter(doc="""
      The plot data declared as a dictionary of arrays or a DataFrame.""")

    layout = param.ObjectSelector(default="column", objects=["column", "row"])

    plot_x = param.String(default="x", doc="""
      The name of the key in the plot_data to use on the x-axis.""")

    plot_y = param.String(default="y", doc="""
      The name of the key in the plot_data to use on the y-axis.""")

    plot_color = param.String(default=BLUE, doc="""
      The color to use in the plot.""")

    plot_type = param.ObjectSelector(default="bar", objects=["line", "step", "area", "bar"], doc="""
      The plot type to render the plot data as.""")

    pos_color = param.String(GREEN, doc="""
      The color used to indicate a positive change.""")

    neg_color = param.String(RED, doc="""
      The color used to indicate a negative change.""")

    title = param.String(doc="""The title or a short description of the card""")

    value = param.Parameter(default='auto', doc="""
      The primary value to be displayed.""")

    value_change = param.Parameter(default='auto', doc="""
      A secondary value. For example the change in percent.""")

    _data_params = ['data']

    _manual_params = ['data']

    _rename = {'data': None, 'selection': None}

    _widget_type = _BkTrendIndicator

    def _get_data(self):
        if self.data is None:
            return None, {self.plot_x: [], self.plot_y: []}
        elif isinstance(self.data, dict):
            return self.data, self.data
        return self.data, ColumnDataSource.from_df(self.data)

    def _init_params(self):
        props = super()._init_params()
        self._processed, self._data = self._get_data()
        props['source'] = ColumnDataSource(data=self._data)
        return props

    def _trigger_auto_values(self):
        trigger = []
        if self.value == 'auto':
            trigger.append('value')
        if self.value_change == 'auto':
            trigger.append('value_change')
        if trigger:
            self.param.trigger(*trigger)

    @updating
    def _stream(self, stream, rollover=None):
        self._trigger_auto_values()
        super()._stream(stream, rollover)

    def _update_cds(self, *events):
        super()._update_cds(*events)
        self._trigger_auto_values()

    def _update_data(self, data):
        if isinstance(data, _BkTrendIndicator):
            return
        super()._update_data(data)

    def _process_param_change(self, msg):
        msg = super()._process_param_change(msg)
        ys = self._data.get(self.plot_y, [])
        if 'value' in msg and msg['value'] == 'auto':
            if len(ys):
                msg['value'] = ys[-1]
            else:
                msg['value'] = 0
        if 'value_change' in msg and msg['value_change'] == 'auto':
            if len(ys) > 1:
                y1, y2 = self._data.get(self.plot_y)[-2:]
                msg['value_change'] = 0 if y1 == 0 else (y2/y1 - 1)
            else:
                msg['value_change'] = 0
        return msg


MARGIN = {
    "text_pane": {"column": (5, 10, 0, 10), "row": (0, 10, 0, 10)},
    "progress": {"column": (0, 10, 5, 10), "row": (12, 10, 0, 10)},
}



class ptqdm(_tqdm):

    def __init__(self, *args, **kwargs):
        self._indicator = kwargs.pop('indicator')
        super().__init__(*args, **kwargs)

    def display(self, msg=None, pos=None, bar_style=None):
        super().display(msg, pos)
        style = self._indicator.text_pane.style or {}
        color = self.colour or 'black'
        self._indicator.text_pane.style = dict(style, color=color)
        if self.total is not None and self.n is not None:
            self._indicator.max = int(self.total) # Can be numpy.int64
            self._indicator.value = int(self.n)
            self._indicator.text = self._to_text(**self.format_dict)
        return True

    def _to_text(self, n, total, **kwargs):
        return self.format_meter(n, total, **{**kwargs, "ncols": 0})

    def close(self):
        super().close()
        if not self.leave:
            self._indicator.reset()
        return _tqdm


class Tqdm(Indicator):

    layout = param.ClassSelector(class_=(Column, Row), precedence=-1, constant=True, doc="""
        The layout for the text and progress indicator.""",)
            
    max = Progress.param.max

    progress = param.ClassSelector(class_=Progress, precedence=-1, doc="""
        The Progress indicator used to display the progress.""",)

    text = param.String(default='', doc="""
        The current tqdm style progress text.""")

    text_pane = param.ClassSelector(class_=Str, precedence=-1, doc="""
        The pane to display the text to.""")
        
    value = Progress.param.value
    value.default = 0

    margin = param.Parameter(default=0, doc="""
        Allows to create additional space around the component. May
        be specified as a two-tuple of the form (vertical, horizontal)
        or a four-tuple (top, right, bottom, left).""")

    width = param.Integer(default=400, bounds=(0, None), doc="""
        The width of the component (in pixels). This can be either
        fixed or preferred width, depending on width sizing policy.""")

    write_to_console = param.Boolean(default=False, doc="""
        Whether or not to also write to the console.""")

    _layouts = {Row: 'row', Column: 'column'}

    _rename = {'value': None, 'min': None, 'max': None, 'text': None}

    def __init__(self, **params):
        layout = params.pop('layout', 'column')
        layout = self._layouts.get(layout, layout) 
        if "text_pane" not in params:
            sizing_mode = 'stretch_width' if layout == 'column' else 'fixed'
            params["text_pane"] = Str(
                None, min_height=20, min_width=280, sizing_mode=sizing_mode,
                margin=MARGIN["text_pane"][layout],
            )
        if "progress" not in params:
            params["progress"] = Progress(
                active=False,
                sizing_mode="stretch_width",
                min_width=100,
                margin=MARGIN["progress"][layout],
            )

        layout_params = {p: params.get(p, getattr(self, p))  for p in Viewable.param}
        if layout == 'row' or layout is Row:
            params['layout'] = Row(
                params['progress'], params['text_pane'], **layout_params
            )
        else:
            params['layout'] = Column(
                params['text_pane'], params['progress'], **layout_params
            )
        super().__init__(**params)

        self.param.watch(self._update_layout, list(Viewable.param))

        self.progress.max = self.max
        self.progress.value = self.value
        self.text_pane.object = self.text

    def _get_model(self, doc, root=None, parent=None, comm=None):
        model = self.layout._get_model(doc, root, parent, comm)
        if root is None:
            root = model
        self._models[root.ref['id']] = (model, parent)
        return model

    def _cleanup(self, root):
        super()._cleanup(root)
        self.layout._cleanup(root)

    def _update_layout(self, *events):
        self.layout.param.set_param(**{event.name: event.new for event in events})

    @param.depends("text", watch=True)
    def _update_text(self):
        if self.text_pane:
            self.text_pane.object = self.text

    @param.depends("value", watch=True)
    def _update_value(self):
        if self.progress:
            self.progress.value = self.value

    @param.depends("max", watch=True)
    def _update_max(self):
        if self.progress:
            self.progress.max = self.max

    def __call__(self, *args, **kwargs):
        kwargs['indicator'] = self
        if not self.write_to_console:
            f = open(os.devnull, 'w')
            kwargs['file'] = f
        return ptqdm(*args, **kwargs)

    __call__.__doc__ = ptqdm.__doc__

    def pandas(self, *args, **kwargs):
        kwargs['indicator'] = self
        if not self.write_to_console and 'file' not in kwargs:
            f = open(os.devnull, 'w')
            kwargs['file'] = f
        return ptqdm.pandas(*args, **kwargs)

    def reset(self):
        """Resets the parameters"""
        self.value = self.param.value.default
        self.text = self.param.text.default
