"""Find your Telegram chat id and send a test message.

1. On Telegram, open @BotFather, send /newbot, follow the prompts, copy the token.
2. Put it in .env:          TELEGRAM_BOT_TOKEN=123456:ABC...
3. Open your new bot on Telegram and send it any message, e.g. "hi".
   (A bot cannot message you until you have messaged it first.)
4. Run:                     python scripts/telegram_setup.py

It prints the chat id. Add that to .env as TELEGRAM_CHAT_ID, run this again,
and you should receive a test message on your phone.
"""

import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smart_property.env import load_dotenv          # noqa: E402


def call(token, method):
    url = f"https://api.telegram.org/bot{token}/{method}"
    with urllib.request.urlopen(url, timeout=20) as r:
        return json.load(r)


def main():
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv()
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise SystemExit("TELEGRAM_BOT_TOKEN is not set. Add it to .env first.")

    try:
        me = call(token, "getMe")["result"]
    except Exception as exc:
        raise SystemExit(f"Token rejected by Telegram: {exc}")
    print(f"bot: @{me['username']} ({me['first_name']})")

    chats = {}
    for update in call(token, "getUpdates").get("result", []):
        msg = update.get("message") or update.get("channel_post") or {}
        chat = msg.get("chat")
        if chat:
            name = chat.get("title") or " ".join(
                filter(None, [chat.get("first_name"), chat.get("last_name")]))
            chats[chat["id"]] = name

    if not chats:
        print("\nNo chats yet. Open the bot on Telegram, send it 'hi', then "
              "run this again.")
        return
    print("\nchats that have messaged the bot:")
    for cid, name in chats.items():
        print(f"  TELEGRAM_CHAT_ID={cid}    ({name})")

    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not chat_id:
        print("\nCopy one of those lines into .env, then run this again to "
              "send a test message.")
        return

    from smart_property.notify import Notifier, TelegramChannel
    notifier = Notifier(TelegramChannel(token, chat_id), log_path=None)
    notifier.test()
    import time
    time.sleep(3)
    last = notifier.history[0] if notifier.history else None
    if last and last["sent"]:
        print(f"\ntest message sent to chat {chat_id}. Check your phone.")
    else:
        print(f"\nsend failed: {last['error'] if last else 'no response'}")


if __name__ == "__main__":
    main()
