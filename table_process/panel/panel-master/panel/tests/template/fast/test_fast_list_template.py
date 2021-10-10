import panel as pn

from bokeh.document import Document
from holoviews import opts

from panel.pane import HoloViews, Markdown
from panel.template.fast.list import FastListDarkTheme, FastListTemplate
from panel.tests.template.fast.test_fast_grid_template import (
    INFO, _create_hvplot, _fast_button_card, _sidebar_items)

ACCENT_COLOR = "purple"

opts.defaults(opts.Ellipse(line_width=3, color=ACCENT_COLOR))

def test_template_theme_parameter():
    template = FastListTemplate(title="Fast", theme="dark")
    # Not '#3f3f3f' which is for the Vanilla theme

    doc = template.server_doc(Document())
    assert doc.theme._json['attrs']['Figure']['background_fill_color']=="#181818"

    assert isinstance(template._get_theme(), FastListDarkTheme)


def test_accepts_colors_by_name():
    template = FastListTemplate(
        accent_base_color="red",
        header_background="green",
        header_color="white",
        header_accent_base_color="blue",
    )
    template._update_vars()

def test_app():
    app = FastListTemplate(
        title="FastListTemplate w. #ORSOME colors",
        site="Panel",
        accent_base_color=ACCENT_COLOR,
        header_background=ACCENT_COLOR,
        header_accent_base_color="#FFFFFF",
        main_layout="",
        shadow=True,
    )
    app.main[:] = [
        Markdown(INFO, sizing_mode="stretch_both"),
        HoloViews(_create_hvplot(), sizing_mode="stretch_both"),
        _fast_button_card(),
        HoloViews(_create_hvplot(), sizing_mode="stretch_both"),
    ]
    app.sidebar.extend(_sidebar_items())

    return app


if __name__.startswith("bokeh"):
    pn.extension(sizing_mode="stretch_width")
    test_app().servable()
