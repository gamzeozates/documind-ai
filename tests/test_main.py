from app.main import get_app_name


def test_get_app_name():
    assert get_app_name() == "DocuMind AI"
