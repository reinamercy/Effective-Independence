"""Makes requests/httpx (and anything built on them: openai, google-genai,
huggingface_hub) trust the OS certificate store instead of only the
certifi-bundled public CA list.

Needed on networks that do TLS interception with a corporate root CA that's
installed in the OS trust store but not in certifi's bundle -- without this,
those libraries raise CERTIFICATE_VERIFY_FAILED even though the connection
is otherwise fine (confirmed: urllib, which uses the OS store by default on
Windows, succeeds on the same host where requests/httpx fail).

Import this before any requests/httpx-based network call. Safe to import
multiple times.
"""
import truststore

truststore.inject_into_ssl()
