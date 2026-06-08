"""Set Space/OAuth env defaults before any test imports ``space/app.py``.

Mirrors the QVerify conftest. The QAgent Space has no OAuth gating (it is fully
open), so importing ``space/app.py`` does not actually take Gradio's OAuth path,
but these harmless defaults keep the test environment identical to QVerify's and
future-proof the suite if a login button is ever added. ``setdefault`` never
overrides values injected by HF at runtime.
"""

import os

os.environ.setdefault("SYSTEM", "spaces")
os.environ.setdefault("SPACE_ID", "Laborator/qagent")
os.environ.setdefault("OAUTH_CLIENT_ID", "ci-dummy")
os.environ.setdefault("OAUTH_CLIENT_SECRET", "ci-dummy")
os.environ.setdefault("OAUTH_SCOPES", "openid profile")
os.environ.setdefault("OPENID_PROVIDER_URL", "https://huggingface.co")
