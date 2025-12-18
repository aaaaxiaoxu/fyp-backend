from __future__ import annotations

from email.message import EmailMessage
import aiosmtplib

from .settings import settings


async def send_verification_email(to_email: str, verify_url: str) -> None:
    # 未配置 SMTP 时：开发阶段直接打印，方便你复制验证
    if not settings.SMTP_HOST or not settings.SMTP_FROM:
        print(f"[DEV] verification url -> {verify_url}")
        return

    msg = EmailMessage()
    msg["From"] = settings.SMTP_FROM
    msg["To"] = to_email
    msg["Subject"] = "Verify your email"
    msg.set_content(f"Click to verify your email:\n\n{verify_url}\n")

    await aiosmtplib.send(
        msg,
        hostname=settings.SMTP_HOST,
        port=settings.SMTP_PORT,
        username=settings.SMTP_USERNAME or None,
        password=settings.SMTP_PASSWORD or None,
        start_tls=settings.SMTP_TLS,
    )
