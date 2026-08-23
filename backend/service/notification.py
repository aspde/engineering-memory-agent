"""Feishu webhook notification — shared send path.

The agent's ``notify_feishu_tool`` and the deterministic event-analysis
notifier both post to the same team Feishu bot webhook, so the HTTP and
payload-building logic lives here once.  The tool adds its own JSON
result envelope on top; the notifier only cares about success/failure.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_feishu_payload(
    message: str,
    msg_type: str = "text",
    title: str | None = None,
) -> dict[str, Any]:
    """Build a Feishu bot webhook payload (text or interactive card)."""
    if msg_type == "interactive":
        return {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {"tag": "plain_text", "content": title or "EMA 巡检通知"},
                    "template": "blue",
                },
                "elements": [
                    {"tag": "markdown", "content": message},
                ],
            },
        }
    return {
        "msg_type": "text",
        "content": {"text": message},
    }


async def send_feishu_message(
    message: str,
    msg_type: str = "text",
    title: str | None = None,
) -> tuple[bool, int | str]:
    """Send *message* to the configured Feishu bot webhook.

    Returns ``(ok, detail)`` — detail is the Feishu response code (int,
    matching the historical ``feishu_status`` contract of
    ``notify_feishu_tool``) on success or an error string on failure.
    Never raises; callers decide whether a failure is worth logging loudly.
    """
    import httpx

    from backend.shared.config import config

    webhook_url = config.feishu_webhook_url
    if not webhook_url:
        return False, "FEISHU_WEBHOOK_URL is not configured"

    payload = build_feishu_payload(message, msg_type=msg_type, title=title)
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(webhook_url, json=payload)
            resp.raise_for_status()
        result = resp.json()
        return True, int(result.get("code", -1))
    except httpx.TimeoutException:
        logger.error("Feishu webhook timed out")
        return False, "Webhook request timed out"
    except Exception as exc:
        logger.error("Feishu webhook failed: %s", exc)
        return False, str(exc)
