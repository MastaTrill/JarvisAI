"""
JarvisAI Messaging Connectors
Slack, Discord, and Telegram bot integrations
"""

import os
import logging
from typing import Dict, Any, Optional, Callable
from abc import ABC, abstractmethod

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)


class MessageConnector(ABC):
    """Base class for messaging connectors"""

    @abstractmethod
    def send_message(self, message: str, **kwargs) -> bool:
        """Send a message"""
        pass

    @abstractmethod
    def send_webhook(self, payload: Dict[str, Any]) -> bool:
        """Handle incoming webhook"""
        pass


class SlackConnector(MessageConnector):
    """Slack messaging connector"""

    def __init__(self, webhook_url: str = None, bot_token: str = None):
        self.webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")
        self.bot_token = bot_token or os.getenv("SLACK_BOT_TOKEN")

    def send_message(self, message: str, channel: str = None, **kwargs) -> bool:
        """Send message to Slack"""
        if not self.webhook_url:
            logger.warning("Slack webhook URL not configured")
            return False

        try:
            import requests

            payload = {"text": message}
            if channel:
                payload["channel"] = channel

            response = requests.post(self.webhook_url, json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to send Slack message: {e}")
            return False

    def send_webhook(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Slack webhook (events, commands, etc.)"""
        event_type = payload.get("type")

        if event_type == "url_verification":
            return {"challenge": payload.get("challenge")}

        if event_type == "event_callback":
            event = payload.get("event", {})
            return {
                "type": event.get("type"),
                "user": event.get("user"),
                "text": event.get("text"),
                "channel": event.get("channel"),
            }

        return {}


class DiscordConnector(MessageConnector):
    """Discord messaging connector"""

    def __init__(self, webhook_url: str = None, bot_token: str = None):
        self.webhook_url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
        self.bot_token = bot_token or os.getenv("DISCORD_BOT_TOKEN")

    def send_message(self, message: str, **kwargs) -> bool:
        """Send message to Discord"""
        if not self.webhook_url:
            logger.warning("Discord webhook URL not configured")
            return False

        try:
            import requests

            payload = {"content": message}

            response = requests.post(self.webhook_url, json=payload, timeout=10)
            return response.status_code in [200, 204]
        except Exception as e:
            logger.error(f"Failed to send Discord message: {e}")
            return False

    def send_webhook(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Discord webhook"""
        return {
            "type": payload.get("type"),
            "guild_id": payload.get("guild_id"),
            "channel_id": payload.get("channel_id"),
            "user": payload.get("member", {}).get("user", {}),
            "message": payload.get("message", {}),
        }


class TelegramConnector(MessageConnector):
    """Telegram messaging connector"""

    def __init__(self, bot_token: str = None):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.api_url = (
            f"https://api.telegram.org/bot{self.bot_token}" if self.bot_token else None
        )

    def send_message(self, message: str, chat_id: str = None, **kwargs) -> bool:
        """Send message to Telegram"""
        if not self.bot_token:
            logger.warning("Telegram bot token not configured")
            return False

        try:
            import requests

            payload = {"text": message, "parse_mode": "Markdown"}
            if chat_id:
                payload["chat_id"] = chat_id

            response = requests.post(
                f"{self.api_url}/sendMessage", json=payload, timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    def send_webhook(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Telegram webhook (updates)"""
        message = payload.get("message", {})
        callback = payload.get("callback_query", {})

        return {
            "update_id": payload.get("update_id"),
            "message": {
                "chat_id": message.get("chat", {}).get("id"),
                "text": message.get("text"),
                "from": message.get("from"),
            },
            "callback": {"data": callback.get("data"), "from": callback.get("from")},
        }


class NotificationManager:
    """Unified notification manager for all connectors"""

    def __init__(self):
        self.slack = SlackConnector()
        self.discord = DiscordConnector()
        self.telegram = TelegramConnector()

    def notify(self, message: str, channels: list = None, **kwargs) -> Dict[str, bool]:
        """Send notification to multiple channels"""
        results = {}

        channels = channels or os.getenv("NOTIFICATION_CHANNELS", "slack").split(",")

        for channel in channels:
            channel = channel.strip().lower()

            if channel == "slack":
                results["slack"] = self.slack.send_message(message, **kwargs)
            elif channel == "discord":
                results["discord"] = self.discord.send_message(message, **kwargs)
            elif channel == "telegram":
                results["telegram"] = self.telegram.send_message(message, **kwargs)

        return results

    def handle_webhook(self, platform: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming webhook from platform"""
        platform = platform.lower()

        if platform == "slack":
            return self.slack.send_webhook(payload)
        elif platform == "discord":
            return self.discord.send_webhook(payload)
        elif platform == "telegram":
            return self.telegram.send_webhook(payload)

        return {}


notification_manager = NotificationManager()
