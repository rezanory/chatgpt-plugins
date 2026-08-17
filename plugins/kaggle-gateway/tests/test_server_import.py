from chatgpt_plugin_kaggle_gateway import server


def test_server_import_does_not_initialize_kaggle_credentials():
    assert server._pool_instance is None
    assert callable(server.main)
