import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

import main  # module-level code runs here (load_dotenv, app creation) but NOT startup


@pytest.fixture(scope="session")
def client():
    # Patch init_db in main's namespace so the on_startup handler is a no-op.
    with patch("main.init_db"), TestClient(main.app) as c:
        yield c
