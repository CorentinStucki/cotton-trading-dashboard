import os
import requests
from typing import List, Dict, Any

BARCHART_GETQUOTE_URL = "https://ondemand.websol.barchart.com/getQuote.json"

class BarchartProvider:
    def __init__(self, api_key: str | None = None, timeout_s: int = 15):
        self.api_key = api_key or os.getenv("BARCHART_API_KEY", "")
        self.timeout_s = timeout_s

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def get_quotes(self, symbols: List[str], fields: List[str] | None = None) -> List[Dict[str, Any]]:
        if not self.api_key:
            raise RuntimeError("Missing BARCHART_API_KEY (set it as env var).")

        payload = {"apikey": self.api_key, "symbols": ",".join(symbols)}
        if fields:
            payload["fields"] = ",".join(fields)

        r = requests.post(BARCHART_GETQUOTE_URL, data=payload, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()

        if "results" not in data:
            raise RuntimeError(f"Unexpected response from Barchart: {data}")

        return data["results"]