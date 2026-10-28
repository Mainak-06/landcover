from streamlit.testing.v1 import AppTest


import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test_app_runs_clean():
    at = AppTest.from_file("../app.py", default_timeout=120).run()
    assert not at.exception, at.exception
    assert len(at.metric) >= 4
    at.sidebar.selectbox[1].select(2020).run()      # from-year change
    assert not at.exception
