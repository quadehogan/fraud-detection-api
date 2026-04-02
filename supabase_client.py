"""
supabase_client.py — Single Supabase client instance for the entire app.

Import `supabase` from this module wherever DB access is needed:
    from supabase_client import supabase
"""
import os

from supabase import Client, create_client

_url: str = os.environ["SUPABASE_URL"]
_key: str = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

supabase: Client = create_client(_url, _key)
