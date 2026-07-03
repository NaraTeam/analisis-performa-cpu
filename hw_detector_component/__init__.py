import os
import streamlit.components.v1 as components

# Create a _RELEASE constant. We'll set this to False while we're developing
# the component, and True when we're ready to package and distribute it.
_RELEASE = True

if not _RELEASE:
    _hw_detector = components.declare_component(
        "hw_detector",
        url="http://localhost:3001",
    )
else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    _hw_detector = components.declare_component("hw_detector", path=parent_dir)

def hw_detector(key=None):
    """Create a new instance of "hw_detector".

    Parameters
    ----------
    key: str or None
        An optional key that uniquely identifies this component. If this is
        None, and the component's arguments are changed, the component will
        be re-mounted in the Streamlit frontend and lose its current state.

    Returns
    -------
    dict or None
        Returns a dict containing 'cores' and 'memory' detected from the browser.
    """
    component_value = _hw_detector(key=key, default=None)
    return component_value
