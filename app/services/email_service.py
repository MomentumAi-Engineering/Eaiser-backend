import os
from dotenv import load_dotenv
import logging
from typing import List, Tuple, Optional, Dict, Any
import base64
import asyncio
from datetime import datetime
from fastapi import HTTPException
import requests

# --------------------------------------------------------------------
# ✅ Environment & Logger Setup
# --------------------------------------------------------------------
load_dotenv()
logger = logging.getLogger("email_service")
logger.setLevel(logging.INFO)

# --------------------------------------------------------------------
# ✅ Shared branded email shell (EAiSER logo header + MomntumAi social footer)
# All outbound emails wrap their content with build_branded_email() for one
# consistent, professional look. Built by string concatenation (NOT f-strings)
# so CSS braces never need escaping.
# --------------------------------------------------------------------
EAISER_LOGO_URL = "https://eaiser.ai/newlogo.png"
SUPPORT_EMAIL = "support@momntumai.com"
# (label, profile URL, white PNG icon URL) — icons8 serves email-safe PNGs.
SOCIAL_LINKS = [
    ("X", "https://x.com/Momntum_Ai", "https://img.icons8.com/ios-filled/50/C8A84E/twitterx.png"),
    ("LinkedIn", "https://www.linkedin.com/company/momntum-ai-us", "https://img.icons8.com/ios-filled/50/C8A84E/linkedin.png"),
    ("Instagram", "https://www.instagram.com/momntum.ai/", "https://img.icons8.com/ios-filled/50/C8A84E/instagram-new.png"),
]


def _social_icons_html() -> str:
    cells = ""
    for name, url, icon in SOCIAL_LINKS:
        cells += (
            '<a href="' + url + '" target="_blank" style="display:inline-block;margin:0 7px;text-decoration:none;">'
            '<img src="' + icon + '" alt="' + name + '" width="22" height="22" '
            'style="width:22px;height:22px;border:0;outline:none;"></a>'
        )
    return cells


def build_branded_email(title: str, inner_html: str, preheader: str = "") -> str:
    """Wrap inner content in the shared EAiSER/MomntumAi email shell."""
    safe_pre = (preheader or title or "EAiSER").replace("<", "").replace(">", "")
    head = (
        '<!DOCTYPE html><html lang="en"><head>'
        '<meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta name="color-scheme" content="light only">'
        '</head>'
        '<body style="margin:0;padding:0;background:#eef1f5;-webkit-font-smoothing:antialiased;">'
    )
    pre = (
        '<div style="display:none;max-height:0;overflow:hidden;opacity:0;color:transparent;">'
        + safe_pre + '</div>'
    )
    wrap_open = (
        '<div style="background:#eef1f5;padding:28px 14px;">'
        '<div style="max-width:600px;margin:0 auto;background:#ffffff;border-radius:16px;'
        'overflow:hidden;box-shadow:0 8px 30px rgba(15,23,42,0.08);border:1px solid #e2e8f0;">'
    )
    header = (
        '<div style="background:linear-gradient(135deg,#0f172a 0%,#1e293b 100%);'
        'padding:34px 24px;text-align:center;border-bottom:3px solid #C8A84E;">'
        '<img src="' + EAISER_LOGO_URL + '" alt="EAiSER" width="60" height="60" '
        'style="width:60px;height:60px;border-radius:15px;border:0;display:inline-block;">'
        '<div style="color:#C8A84E;font-family:Arial,Helvetica,sans-serif;font-size:13px;'
        'font-weight:700;letter-spacing:3px;margin-top:12px;">E A i S E R</div>'
        '<div style="color:#94a3b8;font-family:Arial,Helvetica,sans-serif;font-size:11px;'
        'letter-spacing:1px;margin-top:3px;">by MomntumAi</div>'
        '</div>'
    )
    content = (
        '<div style="padding:34px 30px;font-family:Arial,Helvetica,sans-serif;'
        'color:#334155;font-size:15px;line-height:1.65;">' + inner_html + '</div>'
    )
    footer = (
        '<div style="background:#0f172a;padding:26px 24px;text-align:center;">'
        '<div style="margin-bottom:14px;">' + _social_icons_html() + '</div>'
        '<div style="color:#94a3b8;font-family:Arial,Helvetica,sans-serif;font-size:12px;line-height:1.7;">'
        'EAiSER — AI-powered civic issue reporting, by MomntumAi.<br>'
        '© 2026 MomntumAi · <a href="mailto:' + SUPPORT_EMAIL + '" '
        'style="color:#C8A84E;text-decoration:none;">' + SUPPORT_EMAIL + '</a>'
        '</div></div>'
    )
    wrap_close = '</div></div>'
    tail = '</body></html>'
    return head + pre + wrap_open + header + content + footer + wrap_close + tail


# Reusable inline styles for inner content, so every email looks consistent.
EMAIL_STYLES = {
    "h1": "margin:0 0 16px;font-size:22px;font-weight:700;color:#0f172a;",
    "p": "margin:0 0 14px;font-size:15px;line-height:1.65;color:#475569;",
    "btn": ("display:inline-block;background:#C8A84E;color:#1a1205;text-decoration:none;"
            "font-weight:700;font-size:15px;padding:14px 32px;border-radius:10px;"),
    "btn_wrap": "text-align:center;margin:28px 0 8px;",
    "box": ("background:#f1f5f9;border-left:4px solid #C8A84E;padding:16px 18px;"
            "border-radius:0 8px 8px 0;margin:20px 0;font-size:14px;color:#334155;"),
    "muted": "margin:18px 0 0;font-size:13px;color:#94a3b8;line-height:1.6;",
}

# --------------------------------------------------------------------
# ✅ Core Async Email Sender with Improved Error Handling
# --------------------------------------------------------------------
async def send_email(
    to_email: str,
    subject: str,
    html_content: str,
    text_content: str,
    attachments: Optional[List[str]] = None,
    embedded_images: Optional[List[Tuple[str, str, str]]] = None,
    reply_to: Optional[str] = None,
    retry: bool = True,
    raw_attachments: Optional[List[Dict[str, Any]]] = None,
    cc: Optional[List[str]] = None
) -> bool:
    """
    Sends an email via Postmark API with inline images and attachments.
    Uses asyncio.to_thread to keep the event loop non-blocking.

    `cc` copies every other notified department onto the SAME email thread, so
    all recipients share one conversation and can coordinate via reply-all.
    """
    email_user = os.getenv("EMAIL_USER", "alert@momntumai.com")
    postmark_token = os.getenv("POSTMARK_API_TOKEN")
    dry_run = os.getenv("EMAIL_DRY_RUN", "false").lower() == "true"

    if dry_run:
        logger.info(f"🧪 Email dry-run enabled. Would send to {to_email} subject='{subject}'.")
        return True

    if not postmark_token:
        logger.error("❌ Missing POSTMARK_API_TOKEN environment variable.")
        return False

    # 🔒 Validate the recipient before hitting Postmark. Sending to malformed /
    # junk addresses drives up the bounce rate, which is what got the stream
    # paused. Skip anything that isn't a syntactically valid single address.
    import re
    addr = (to_email or "").strip()
    if not re.match(r"^[^@\s,;]+@[^@\s,;]+\.[^@\s,;]+$", addr):
        logger.warning(f"⚠️ Skipping send — invalid recipient email: {to_email!r}")
        return False
    to_email = addr

    # 📋 Clean the CC list: keep only valid addresses, drop duplicates and the
    # primary recipient so nobody is listed twice on the thread.
    cc_clean: List[str] = []
    if cc:
        seen = {to_email.lower()}
        for c in cc:
            c = (c or "").strip()
            if c and c.lower() not in seen and re.match(r"^[^@\s,;]+@[^@\s,;]+\.[^@\s,;]+$", c):
                seen.add(c.lower())
                cc_clean.append(c)

    # Build Postmark payload
    payload = {
        "From": email_user,
        "To": to_email,
        "Subject": subject,
        "HtmlBody": html_content,
        "TextBody": text_content,
        "Attachments": []
    }

    if cc_clean:
        payload["Cc"] = ", ".join(cc_clean)

    if reply_to:
        payload["ReplyTo"] = reply_to

    # Add inline images to Attachments
    if embedded_images:
        for cid, base64_data, mime_type in embedded_images:
            try:
                if not base64_data:
                    logger.warning(f"⚠️ Skipping empty embedded image {cid}")
                    continue
                payload["Attachments"].append({
                    "Name": f"{cid}.{mime_type.split('/')[-1]}",
                    "Content": base64_data,
                    "ContentType": mime_type,
                    "ContentID": f"cid:{cid}"
                })
                logger.debug(f"🖼️ Embedded image {cid} added.")
            except Exception as e:
                logger.error(f"⚠️ Failed to prepare inline image {cid}: {e}")

    # Add attachments to Attachments
    if attachments:
        for file_path in attachments:
            try:
                with open(file_path, "rb") as f:
                    data = f.read()
                # Postmark requires base64 content
                encoded = base64.b64encode(data).decode()
                payload["Attachments"].append({
                    "Name": os.path.basename(file_path),
                    "Content": encoded,
                    "ContentType": "application/octet-stream"
                })
                logger.debug(f"📎 Attached file: {file_path}")
            except Exception as e:
                logger.error(f"⚠️ Failed to prepare attachment {file_path}: {e}")

    # Add raw pre-encoded attachments
    if raw_attachments:
        for attachment in raw_attachments:
            # Postmark expects keys: Name, Content, ContentType
            if all(k in attachment for k in ["Name", "Content", "ContentType"]):
                content_len = len(attachment.get("Content", ""))
                logger.info(f"📎 Adding raw attachment: {attachment.get('Name')} ({attachment.get('ContentType')}) - Size: {content_len} chars")
                payload["Attachments"].append(attachment)
                logger.debug(f"📎 Added raw attachment: {attachment.get('Name')}")

    def _do_send():
        url = "https://api.postmarkapp.com/email"
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-Postmark-Server-Token": postmark_token
        }
        cc_note = f" CC {len(cc_clean)} dept(s)" if cc_clean else ""
        logger.info(f"📤 Sending Postmark email FROM {email_user} TO {to_email}{cc_note}")
        return requests.post(url, headers=headers, json=payload, timeout=15)

    try:
        response = await asyncio.to_thread(_do_send)
        
        if response.status_code == 200:
            logger.info(f"✅ Email successfully sent to {to_email} via Postmark")
            return True
        else:
            try:
                error_data = response.json()
                msg = error_data.get('Message', 'Unknown Postmark error')
            except:
                msg = response.text
            
            logger.warning(f"⚠️ Postmark API returned {response.status_code}: {msg}")
            
            if retry and response.status_code != 401:
                logger.info("🔁 Retrying once after 2 seconds...")
                await asyncio.sleep(2)
                return await send_email(
                    to_email, subject, html_content, text_content,
                    attachments, embedded_images,
                    reply_to=reply_to, retry=False,
                    raw_attachments=raw_attachments,
                    cc=cc
                )
            return False

    except Exception as e:
        logger.error(f"❌ Unexpected error sending email to {to_email} via Postmark: {e}")
        return False


# --------------------------------------------------------------------
# ✅ Synchronous Wrapper for Celery/Testing
# --------------------------------------------------------------------
def send_email_sync(
    to_email: str,
    subject: str,
    html_content: str,
    text_content: str,
    attachments: Optional[List[str]] = None,
    embedded_images: Optional[List[Tuple[str, str, str]]] = None
) -> bool:
    """Run async send_email synchronously (for local testing or Celery)."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(
            send_email(to_email, subject, html_content, text_content, attachments, embedded_images)
        )
    except Exception as e:
        logger.error(f"❌ Sync wrapper failed: {e}")
        return False
    finally:
        loop.close()


# --------------------------------------------------------------------
# ✅ EmailService Class for Issue Notifications
# --------------------------------------------------------------------
class EmailService:
    async def send_issue_notification(self, authorities: List[Dict[str, Any]], issue_data: Dict[str, Any], issue_id: str) -> bool:
        """Send AI-detected issue notification emails to relevant authorities."""
        try:
            subject = f"New Issue Report #{issue_data.get('report_id', issue_id)}"
            text_content = (
                f"Issue ID: {issue_id}\n"
                f"Type: {issue_data.get('issue_type')}\n"
                f"Severity: {issue_data.get('severity')}\n"
                f"Description: {issue_data.get('description', 'N/A')}\n"
                f"Location: {issue_data.get('address', 'N/A')} (ZIP: {issue_data.get('zip_code', 'N/A')})\n"
            )
            html_content = (
                f"<h3>🛰️ New Issue Report Detected</h3>"
                f"<p><strong>Issue ID:</strong> {issue_id}</p>"
                f"<p><strong>Type:</strong> {issue_data.get('issue_type')}</p>"
                f"<p><strong>Severity:</strong> {issue_data.get('severity')}</p>"
                f"<p><strong>Description:</strong> {issue_data.get('description', 'N/A')}</p>"
                f"<p><strong>Location:</strong> {issue_data.get('address', 'N/A')} (ZIP: {issue_data.get('zip_code', 'N/A')})</p>"
            )

            sent_any = False
            for auth in authorities:
                to_email = auth.get('email') or auth.get('contact_email')
                if not to_email:
                    logger.warning(f"⚠️ Skipping authority without email: {auth}")
                    continue
                ok = await send_email(to_email, subject, html_content, text_content)
                sent_any = sent_any or ok

            logger.info(f"📬 Issue email notifications sent: {sent_any}")
            return sent_any

        except Exception as e:
            logger.error(f"❌ Issue notification failed: {e}", exc_info=True)
            return False


def get_email_service() -> EmailService:
    """Return a reusable instance of EmailService."""
    return EmailService()


# --------------------------------------------------------------------
# ✅ Fallback report body (so authority emails are never empty)
# --------------------------------------------------------------------
def _build_fallback_report_body(report: Dict[str, Any]) -> str:
    """
    Compose a readable report body from whatever fields exist on the report.

    Some report paths (the V3 engine, no-issue / manual-review reports, and
    pre-check rejections) never generate the AI's pre-formatted `formatted_report`
    string — which used to leave the authority email body completely blank. This
    rebuilds a clear body from issue_overview / template_fields / detailed_analysis
    so the recipient always sees the incident details.
    """
    ov = report.get("issue_overview", {}) or {}
    tf = report.get("template_fields", {}) or {}
    da = report.get("detailed_analysis", {}) or {}
    ai = report.get("ai_evaluation", {}) or {}

    issue_type = ov.get("type") or report.get("issue_type") or "Reported issue"
    priority = tf.get("priority") or ov.get("severity") or "N/A"
    report_id = tf.get("oid") or report.get("_id") or "N/A"
    timestamp = tf.get("timestamp") or ""
    address = tf.get("address") or report.get("address") or "Location not provided"
    zip_code = str(tf.get("zip_code") or report.get("zip_code") or "")
    map_link = tf.get("map_link") or ""

    # Best available narrative, in priority order.
    summary = (
        ov.get("summary_explanation")
        or ov.get("detailed_description")
        or ov.get("summary")
        or report.get("scene_description")
        or report.get("issue_summary")
        or report.get("description")
        or ""
    ).strip()

    issue_detected = ai.get("issue_detected", True)
    type_blank = str(issue_type).strip().lower() in ("none", "", "no issue", "reported issue")

    lines = ["EAiSER Civic Incident Report", ""]
    lines.append(f"Issue Type: {issue_type}")
    lines.append(f"Priority: {priority}")
    if str(report_id) != "N/A":
        lines.append(f"Report ID: {report_id}")
    if timestamp:
        lines.append(f"Reported: {timestamp}")
    lines.append("")

    location = address + (f" {zip_code}" if zip_code and zip_code not in address else "")
    lines.append(f"Location: {location}")
    if map_link:
        lines.append(f"Map: {map_link}")
    lines.append("")

    if not issue_detected or type_blank:
        lines.append(
            "EAiSER's AI could not clearly detect a public infrastructure issue in the "
            "submitted photo. This report has been forwarded for manual review so the "
            "right team can take a closer look."
        )
        lines.append("")

    if summary:
        lines.append("Summary:")
        lines.append(summary)
        lines.append("")

    root = (da.get("root_causes") or "").strip()
    cons = (da.get("potential_consequences_if_ignored") or "").strip()
    if root and root.lower() not in ("not specified.", "n/a", "none"):
        lines.append("Possible cause:")
        lines.append(root)
        lines.append("")
    if cons and cons.lower() not in ("n/a", "none"):
        lines.append("If left unaddressed:")
        lines.append(cons)
        lines.append("")

    lines.append("Please review and let us know if another department is better suited to handle this.")
    return "\n".join(lines).strip()


# --------------------------------------------------------------------
# ✅ Send AI-Formatted Alert to Authorities (EAiSER Alert)
# --------------------------------------------------------------------
async def send_formatted_ai_alert(report: Dict[str, Any], background: bool = True) -> Dict[str, Any]:
    """
    Send the AI-generated formatted EAiSER alert to authorities.
    Uses the 'formatted_report' field from the AI report.
    """
    try:
        env = os.getenv("ENV", "development").lower()
        dry_run = os.getenv("EMAIL_DRY_RUN", "false").lower() == "true"
        formatted_content = report.get("formatted_report", "")
        # 🛟 Never send an empty body. Some report paths (V3 engine, no-issue /
        # manual-review, pre-check rejections) don't produce `formatted_report`,
        # which left the email blank — rebuild a body from the report fields.
        if not formatted_content or not formatted_content.strip():
            logger.warning("⚠️ formatted_report missing/empty — building fallback body from report fields.")
            formatted_content = _build_fallback_report_body(report)
        issue_type = report.get("issue_overview", {}).get("type", "Issue")
        report_id = report.get("template_fields", {}).get("oid", "N/A")
        priority = report.get("template_fields", {}).get("priority", "N/A")
        subject = f"EAiSER Alert – {issue_type} (Priority: {priority}, ID: {report_id})"

        authorities = report.get("responsible_authorities_or_parties") or report.get("available_authorities") or []
        recipients = [a.get("email") for a in authorities if isinstance(a, dict) and a.get("email")]

        if not recipients:
            logger.warning("⚠️ No recipients found in AI report.")
            return {"status": "no_recipients", "recipients": []}

        # 📋 "Departments notified" block — list every department copied on this
        # email so each recipient can see who else is on the thread.
        notified = [a for a in authorities if isinstance(a, dict) and a.get("email")]
        dept_items = "".join(
            f"<li style='margin-bottom:4px;'><strong>{a.get('name', 'Department')}</strong>"
            f"{(' — ' + str(a.get('type', '')).replace('_', ' ').title()) if a.get('type') else ''}</li>"
            for a in notified
        )
        departments_html = f"""
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:20px;margin-bottom:25px;">
            <h3 style="margin:0 0 8px;color:#1e293b;font-size:15px;">Departments Notified</h3>
            <p style="color:#64748b;font-size:13px;margin:0 0 12px;">Every department below has been copied on this email <strong>in the spirit of collaboration</strong>, so you can coordinate directly by replying to this thread.</p>
            <ul style="margin:0 0 14px;padding-left:18px;color:#334155;font-size:13px;line-height:1.6;">{dept_items}</ul>
            <p style="color:#64748b;font-size:13px;margin:0;">If a different department or office is better suited to handle this, or there's a better way for us to route reports like this, please just reply and let us know — we're happy to adjust.</p>
        </div>
        """
        departments_text = (
            "Departments notified — all copied on this email in the spirit of collaboration, "
            "so you can coordinate directly by replying to this thread:\n"
            + "\n".join(
                f" - {a.get('name', 'Department')}" + (f" ({a.get('type')})" if a.get('type') else "")
                for a in notified
            )
            + "\n\nIf a different department is better suited to handle this, or there's a better way "
            "for us to route reports like this, please reply and let us know — we're happy to adjust."
        )

        # Safe defaults so the body always includes the departments list even if
        # the richer template below is skipped.
        final_html = build_branded_email(
            f"EAiSER Alert – {issue_type}",
            departments_html + formatted_content.replace("\n", "<br>"),
            preheader=f"EAiSER Alert – {issue_type} (Priority: {priority})",
        )
        final_text = f"{departments_text}\n\n{formatted_content}"

        # Only skip if dry_run is explicitly enabled
        if dry_run:
            logger.info(
                f"🧪 Email dry-run active. Skipping send to {len(recipients)} recipients"
            )
            return {"status": "dry_run", "recipients": recipients}

        # -------------------------------------------------------------
        # 📧 Automated Email Communication Note
        # -------------------------------------------------------------
        try:
            # We need the real DB ID
            real_id = report.get("template_fields", {}).get("oid") or report.get("_id")
            
            if real_id:
                button_html = f"""
                <hr style="border: 0; border-top: 1px solid #e5e7eb; margin: 30px 0;">
                <div style="text-align: center; background-color: #f8fafc; padding: 25px; border-radius: 12px; border: 1px solid #e2e8f0;">
                    <h3 style="margin-top: 0; color: #1e293b; font-size: 18px;">📧 Official Communication</h3>
                    <p style="color: #64748b; margin-bottom: 25px; font-size: 14px;">To coordinate with the citizen regarding this incident, please <strong>reply directly to this email</strong>.</p>
                    
                    <p style="margin-top: 15px; font-size: 11px; color: #94a3b8; line-height: 1.5;">
                        <strong>Note:</strong> Your reply will be automatically routed to the reporter. You can request additional details or provide updates via this thread.<br>
                        Issue ID: #{real_id} • Automated Routing Active
                    </p>
                </div>
                """

                # Construct final HTML and Text bodies
                formatted_html = formatted_content.replace('\n', '<br>')
                alert_badge = (
                    '<div style="text-align:center;margin-bottom:24px;">'
                    '<div style="display:inline-block;padding:4px 12px;background:rgba(200,168,78,0.12);border:1px solid rgba(200,168,78,0.3);border-radius:50px;font-size:11px;font-weight:700;color:#b45309;text-transform:uppercase;letter-spacing:1px;">'
                    'Official Intelligence Transmission'
                    '</div></div>'
                )
                report_body = (
                    '<div style="background:#f8fafc;border-radius:16px;padding:22px;border:1px solid #f1f5f9;margin-bottom:24px;font-size:15px;color:#334155;line-height:1.8;">'
                    + formatted_html +
                    '</div>'
                )
                inner_html = alert_badge + departments_html + report_body + button_html
                final_html = build_branded_email(
                    f"EAiSER Smart Alert – {issue_type}",
                    inner_html,
                    preheader=f"EAiSER Alert – {issue_type} (Priority: {priority}, ID: {report_id})",
                )

                final_text = f"{departments_text}\n\n{formatted_content}\n\nREPLY TO THIS EMAIL to communicate with the reporter (Issue ID: #{real_id})."

        except Exception as token_error:
            logger.error(f"⚠️ Failed to generate authority token: {token_error}")
            final_html = build_branded_email(
                f"EAiSER Alert – {issue_type}",
                departments_html + formatted_content.replace("\n", "<br>"),
                preheader=f"EAiSER Alert – {issue_type} (Priority: {priority})",
            )
            final_text = f"{departments_text}\n\n{formatted_content}"

        # Send ONE email so every department shares a single thread: the first
        # recipient goes in To, the rest are CC'd. Reply-all keeps everyone — and
        # the reporter (via ReplyTo) — in the same conversation.
        inbound_email = os.getenv("POSTMARK_INBOUND_EMAIL", "reports@inbound.eaiser.ai")
        primary, cc_emails = recipients[0], recipients[1:]

        async def _send():
            return await send_email(primary, subject, final_html, final_text, reply_to=inbound_email, cc=cc_emails)

        if background:
            asyncio.create_task(_send())
            logger.info(f"📤 Dispatched EAiSER Alert thread — To {primary}, Cc {len(cc_emails)} dept(s) (background mode)")
            return {"status": "dispatched", "recipients": recipients}
        else:
            ok = await _send()
            return {"status": "completed", "sent": recipients if ok else [], "failed": [] if ok else recipients}

    except Exception as e:
        logger.error(f"❌ send_formatted_ai_alert failed: {str(e)}", exc_info=True)
        return {"status": "error", "error": str(e)}

# --------------------------------------------------------------------
# ✅ Notify User Status Change
# --------------------------------------------------------------------
async def notify_user_status_change(user_email: str, issue_id: str, status: str, notes: Optional[str] = None) -> bool:
    """
    Notify the user that their report status has changed (Approved/Rejected) using new templates.
    """
    try:
        from services.mongodb_optimized_service import get_optimized_mongodb_service
        mongo = await get_optimized_mongodb_service()
        issue = await mongo.get_issue_by_id(issue_id) if mongo else {}
        
        # User name fallback
        user_name = "User"
        if issue and issue.get("reporter_name"):
            user_name = issue.get("reporter_name")
        elif user_email:
            user_name = user_email.split("@")[0].capitalize()

        # Issue Name / Description
        issue_type = "Issue"
        if issue and issue.get("issue_type"):
            issue_type = issue.get("issue_type").replace("_", " ").title()
        elif issue and issue.get("category"):
            issue_type = issue.get("category").replace("_", " ").title()
        
        # Subject prefix
        subject = f"Your EAiSER report – {issue_type} - Issue ID: {issue_id}"
        
        # Dashboard URL
        frontend_url = os.getenv("FRONTEND_URL", "https://www.eaiser.ai")
        dashboard_url = f"{frontend_url}/dashboard"

        # Branded shell inner content — greeting + (status block) + closing + CTA.
        base_html_start = (
            '<div style="' + EMAIL_STYLES["h1"] + '">Hi ' + str(user_name) + ',</div>'
        )

        base_html_end = (
            '<p style="' + EMAIL_STYLES["p"] + '">If you have any questions or additional information, please reply to this email and our support team will follow up as soon as possible.</p>'
            '<p style="' + EMAIL_STYLES["p"] + 'margin-top:24px;font-weight:600;color:#1e293b;">Best regards,<br>The EAiSER Team</p>'
            '<div style="' + EMAIL_STYLES["btn_wrap"] + '">'
            '<a href="' + dashboard_url + '" style="' + EMAIL_STYLES["btn"] + '">Back to Dashboard</a>'
            '</div>'
        )

        if status == "approved":
            # Extract authorities
            report = issue.get("report") or {}
            auths_data = report.get("responsible_authorities_or_parties") or report.get("available_authorities", [])
            auth_names = []
            for a in auths_data:
                if isinstance(a, dict) and a.get("name"):
                    auth_names.append(a.get("name"))
                elif isinstance(a, str):
                    auth_names.append(a)
            
            # Format authorities list e.g., "Authority X and Authority Y" or "the relevant authorities"
            if auth_names:
                if len(auth_names) > 1:
                    auth_display = ", ".join(auth_names[:-1]) + f" and {auth_names[-1]}"
                else:
                    auth_display = auth_names[0]
            else:
                auth_display = "the relevant authorities"

            approved_box = (
                '<div style="background:#f0fdf4;border-left:4px solid #22c55e;padding:16px 18px;border-radius:0 8px 8px 0;margin:20px 0;font-size:14px;color:#334155;">'
                '<strong>Issue ID:</strong> ' + str(issue_id) + '<br><br>'
                'Based on the information shared, we have successfully provided the details to <strong>' + str(auth_display) + '</strong> for further action.'
                '</div>'
            )
            inner_html = (
                base_html_start
                + '<p style="' + EMAIL_STYLES["p"] + '">We have completed the review of your report.</p>'
                + approved_box
                + base_html_end
            )
            html_content = build_branded_email(
                "EAiSER Report Update",
                inner_html,
                preheader="We have completed the review of your report.",
            )

            text_content = (
                f"Hi {user_name},\n\n"
                f"We have completed the review of your EAiSER report (Issue ID: {issue_id}).\n\n"
                f"Based on the information shared, we have provided the details to {auth_display}.\n\n"
                "If you have any questions or additional information, please reply to this email and our support team will follow up as soon as possible.\n\n"
                "Best regards,\n"
                "The EAiSER Team\n\n"
                f"Back to Dashboard: {dashboard_url}"
            )
            subject += "."  # The prompt has a dot at the end of the approved subject
        else:
            # Declined / Rejected
            declined_box = (
                '<div style="background:#fef2f2;border-left:4px solid #ef4444;padding:16px 18px;border-radius:0 8px 8px 0;margin:20px 0;font-size:14px;color:#334155;">'
                '<strong>Issue ID:</strong> ' + str(issue_id) + '<br><br>'
                'At this time, we have not identified any issues that require notification to authorities. '
                '</div>'
            )
            inner_html = (
                base_html_start
                + '<p style="' + EMAIL_STYLES["p"] + '">We have completed the review of your report.</p>'
                + declined_box
                + base_html_end
            )
            html_content = build_branded_email(
                "EAiSER Report Update",
                inner_html,
                preheader="We have completed the review of your report.",
            )

            text_content = (
                f"Hi {user_name},\n\n"
                f"We have completed the review of your EAiSER report (Issue ID: {issue_id}).\n\n"
                "We have not identified any issues that require notification to authorities at this time.\n\n"
                "If you have any questions or further details to share, please reply to this email and our support team will follow up as soon as possible.\n\n"
                "Best regards,\n"
                "The EAiSER Team\n\n"
                f"Back to Dashboard: {dashboard_url}"
            )

        return await send_email(user_email, subject, html_content, text_content)
    except Exception as e:
        logger.error(f"Failed to notify user {user_email}: {e}", exc_info=True)
        return False


# --------------------------------------------------------------------
# ✅ Admin Welcome Email (Animated & Professional)
# --------------------------------------------------------------------

ADMIN_DASHBOARD_URL = "https://www.eaiser.ai/admin"

async def send_admin_welcome_email(
    admin_email: str,
    admin_name: str,
    role: str,
    temporary_password: str,
    created_by: str
) -> bool:
    """
    Sends a professional, animated welcome email to newly created admins.
    """
    try:
        # ----------------------------
        # Role Permissions Mapping
        # ----------------------------
        role_permissions = {
            "super_admin": "Full system access — manage admins, assign issues, approve or decline reports.",
            "admin": "Manage team members, assign issues, approve or decline reports.",
            "team_member": "Handle assigned issues and review reports.",
            "viewer": "Read-only access to dashboards and reports."
        }

        permissions_text = role_permissions.get(
            role,
            "Standard administrative access."
        )

        subject = f"Welcome to EAiSER — {role.replace('_', ' ').title()} Access Granted"

        # ----------------------------
        # HTML EMAIL (branded shell)
        # ----------------------------
        role_title = role.replace('_', ' ').title()
        credential_card = (
            '<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:16px;padding:24px 26px;margin:24px 0;">'
            '<h3 style="margin:0 0 16px;font-size:13px;color:#64748b;text-transform:uppercase;letter-spacing:1px;font-weight:700;">&#128272; Security Credentials</h3>'
            '<div style="margin-bottom:14px;">'
            '<span style="font-size:11px;color:#94a3b8;display:block;margin-bottom:4px;text-transform:uppercase;letter-spacing:0.5px;font-weight:600;">Access Email</span>'
            '<span style="font-size:16px;color:#1e293b;font-weight:600;">' + str(admin_email) + '</span>'
            '</div>'
            '<div>'
            '<span style="font-size:11px;color:#94a3b8;display:block;margin-bottom:4px;text-transform:uppercase;letter-spacing:0.5px;font-weight:600;">Temporary Access Token</span>'
            '<div style="background:#ffffff;border:1px dashed #cbd5e1;padding:12px 18px;border-radius:10px;font-family:Monaco,Consolas,monospace;font-size:16px;color:#0f172a;font-weight:700;display:table;margin-top:6px;letter-spacing:1px;">' + str(temporary_password) + '</div>'
            '</div>'
            '<p style="margin:15px 0 0;font-size:12px;color:#ef4444;font-weight:600;">&#9888; For security compliance, you must update this password upon initial authentication.</p>'
            '</div>'
        )
        permissions_box = (
            '<div style="background:#fffbeb;border-left:4px solid #C8A84E;padding:18px 20px;border-radius:0 12px 12px 0;margin:24px 0;">'
            '<h4 style="margin:0 0 6px;color:#1e293b;font-size:15px;font-weight:700;">&#128737; Privileged Scope</h4>'
            '<p style="margin:0;font-size:14px;color:#64748b;line-height:1.6;">' + str(permissions_text) + '</p>'
            '</div>'
        )
        inner_html = (
            '<h1 style="' + EMAIL_STYLES["h1"] + '">System Access Granted</h1>'
            '<div style="display:inline-block;padding:5px 14px;background:#fffbeb;border:1px solid #fde68a;color:#b45309;border-radius:50px;font-size:12px;font-weight:700;margin-bottom:18px;text-transform:uppercase;letter-spacing:0.5px;">Role: ' + role_title + '</div>'
            '<p style="' + EMAIL_STYLES["p"] + '">Hello <strong>' + str(admin_name) + '</strong>,</p>'
            '<p style="' + EMAIL_STYLES["p"] + '">You have been officially onboarded to the EAiSER administrative network by <strong>' + str(created_by) + '</strong>. Your account is now active and ready for deployment.</p>'
            + credential_card
            + permissions_box
            + '<div style="' + EMAIL_STYLES["btn_wrap"] + '">'
            '<a href="' + ADMIN_DASHBOARD_URL + '" style="' + EMAIL_STYLES["btn"] + '">Launch Admin Console &rarr;</a>'
            '</div>'
            '<p style="' + EMAIL_STYLES["muted"] + 'text-align:center;">Welcome to the team. Let\'s make a difference.</p>'
        )
        html_content = build_branded_email(
            "Admin access granted",
            inner_html,
            preheader="Your EAiSER admin account is active — credentials inside.",
        )

        # ----------------------------
        # TEXT EMAIL (Fallback)
        # ----------------------------
        text_content = f"""
Welcome to EAiSER

Hi {admin_name},

You have been onboarded by {created_by} as a {role.replace('_', ' ').title()}.

Your Login Credentials:
-----------------------
Email: {admin_email}
Temporary Password: {temporary_password}

(Please change your password after first login)

Your Permissions:
-----------------
{permissions_text}

Access Admin Dashboard:
{ADMIN_DASHBOARD_URL}

— EAiSER Platform
"""

        # ----------------------------
        # SEND EMAIL
        # ----------------------------
        return await send_email(
            to_email=admin_email,
            subject=subject,
            html_content=html_content,
            text_content=text_content
        )

    except Exception as e:
        logger.error(f"❌ Failed to send admin welcome email to {admin_email}: {e}")
        return False


# --------------------------------------------------------------------
# ✅ User Verification Email (Sleek & Actionable)
# --------------------------------------------------------------------

async def send_verification_email(user_email: str, user_name: str, token: str) -> bool:
    """
    Sends a verification email to new users.
    """
    try:
        frontend_url = os.getenv("FRONTEND_URL", "https://www.eaiser.ai")
            
        verification_link = f"{frontend_url}/verify-email?token={token}"
        subject = "Verify your EAiSER AI Account"

        inner_html = (
            '<h1 style="' + EMAIL_STYLES["h1"] + '">Welcome to the Mission, ' + str(user_name) + '!</h1>'
            '<p style="' + EMAIL_STYLES["p"] + '">Please verify your email address to activate your account and start reporting civic issues with AI precision.</p>'
            '<div style="' + EMAIL_STYLES["btn_wrap"] + '">'
            '<a href="' + verification_link + '" style="' + EMAIL_STYLES["btn"] + '">Verify My Account &rarr;</a>'
            '</div>'
            '<div style="' + EMAIL_STYLES["box"] + 'word-break:break-all;">'
            'Or copy this link into your browser:<br/>' + verification_link + '</div>'
            '<p style="' + EMAIL_STYLES["muted"] + 'text-align:center;">This link will expire in 24 hours.</p>'
        )
        html_content = build_branded_email(
            "Verify your account",
            inner_html,
            preheader="Verify your email to activate your EAiSER AI account.",
        )
        text_content = f"Welcome to EAiSER AI, {user_name}!\n\nPlease verify your email: {verification_link}"
        
        return await send_email(user_email, subject, html_content, text_content)
    except Exception as e:
        logger.error(f"Failed to send verification email: {e}")
        return False

async def send_user_welcome_email(user_email: str, user_name: str) -> bool:
    """
    Sends a premium, high-converting welcome email to new users.
    Focuses on rich aesthetics, animations, and community impact.
    """
    try:
        subject = "Welcome to EAiSER AI – Your Journey to a Smarter Community Begins 🚀"

        # ----------------------------
        # HTML EMAIL (branded shell)
        # ----------------------------
        def _feature(icon: str, title: str, desc: str) -> str:
            return (
                '<div style="background:#f1f5f9;border-radius:12px;padding:18px 20px;margin-bottom:14px;">'
                '<div style="font-weight:700;color:#1e293b;font-size:16px;margin-bottom:4px;">'
                + icon + ' ' + title + '</div>'
                '<div style="font-size:14px;color:#64748b;line-height:1.5;">' + desc + '</div>'
                '</div>'
            )

        inner_html = (
            '<h1 style="' + EMAIL_STYLES["h1"] + '">Welcome to the inner circle, ' + str(user_name) + '!</h1>'
            '<p style="' + EMAIL_STYLES["p"] + '">You\'ve just unlocked the most powerful tool for community transformation. EAiSER isn\'t just an app&mdash;it\'s an AI-driven mission to eliminate civic hazards and build a more responsive world, one report at a time.</p>'
            + _feature("&#128247;", "Impact via Imagery", "Snap a photo and let our Neural Engine handle the rest. We identify complexity, severity, and root causes instantly.")
            + _feature("&#9889;", "Autonomous Routing", "Say goodbye to bureaucratic hurdles. EAiSER routes your data directly to the precise authority responsible for action.")
            + _feature("&#128200;", "Real-time Visibility", "Monitor the status of your reports on a live dashboard. See exactly when officials receive, review, and resolve issues.")
            + '<div style="' + EMAIL_STYLES["btn_wrap"] + '">'
            '<a href="https://www.eaiser.ai/dashboard" style="' + EMAIL_STYLES["btn"] + '">Initialize My Dashboard</a>'
            '</div>'
            '<p style="' + EMAIL_STYLES["muted"] + 'text-align:center;">Together, we\'re building the future of civic technology.</p>'
        )
        html_content = build_branded_email(
            "Welcome to EAiSER AI",
            inner_html,
            preheader="Your journey to a smarter community begins now.",
        )
        # ----------------------------
        # TEXT FALLBACK
        # ----------------------------
        text_content = f"""
Welcome to EAiSER AI, {user_name}!

You've just joined the most powerful network for community transformation.

What's Next?
1. Snap & Report: Our Neuro Engine analyzes issues instantly.
2. Autonomous Routing: We send data directly to the right authorities.
3. Real-time Tracking: Monitor progress on your live dashboard.

Get Started: https://www.eaiser.ai/dashboard

Together, let's build smarter communities.
- Team EAiSER
"""
        return await send_email(user_email, subject, html_content, text_content)

    except Exception as e:
        logger.error(f"❌ Failed to send user welcome email to {user_email}: {e}")
        return False
# --------------------------------------------------------------------
# ✅ Password Reset Email
# --------------------------------------------------------------------

async def send_password_reset_email(email: str, token: str, is_admin: bool = False) -> bool:
    """
    Sends a secure password reset email with a 15-minute expiry warning.

    When ``is_admin`` is True the reset link points to the admin console
    (ADMIN_URL) instead of the public site, since admins and users live in
    separate apps with separate reset pages.
    """
    try:
        # Determine Frontend URL — admins reset on the admin subdomain.
        if is_admin:
            frontend_url = os.getenv("ADMIN_URL", "https://admin.eaiser.ai")
        else:
            frontend_url = os.getenv("FRONTEND_URL", "https://www.eaiser.ai")
        if frontend_url.endswith("/"):
            frontend_url = frontend_url[:-1]

        reset_link = f"{frontend_url}/reset-password?token={token}"
        subject = "Reset Your EAiSER Password"

        inner_html = (
            '<h1 style="' + EMAIL_STYLES["h1"] + '">Secure Password Reset</h1>'
            '<p style="' + EMAIL_STYLES["p"] + '">We received a request to reset your EAiSER account password. Click the button below to choose a new one.</p>'
            '<p style="' + EMAIL_STYLES["p"] + 'font-weight:600;color:#dc2626;">This link expires in 15 minutes.</p>'
            '<div style="' + EMAIL_STYLES["btn_wrap"] + '">'
            '<a href="' + reset_link + '" style="' + EMAIL_STYLES["btn"] + '">Reset My Password &rarr;</a>'
            '</div>'
            '<div style="' + EMAIL_STYLES["box"] + '">If you didn\'t request this, you can safely ignore this email. Your password will remain unchanged.</div>'
        )
        html_content = build_branded_email(
            "Reset your password",
            inner_html,
            preheader="Reset your EAiSER password — link expires in 15 minutes.",
        )
        text_content = f"Reset your EAiSER password: {reset_link}\n\nThis link will expire in 15 minutes."

        return await send_email(email, subject, html_content, text_content)
    except Exception as e:
        logger.error(f"Failed to send password reset email to {email}: {e}")
        return False

async def send_tos_email(email: str, name: str) -> bool:
    """Send a copy of the Terms of Service to the user upon acceptance."""
    # 🔒 Kill-switch: the bulk TOS/welcome email (sent on every signup) was the
    # source of the bot-driven spam spike that got the Postmark stream paused.
    # Set TOS_EMAIL_ENABLED=false to instantly stop it if a bot wave hits.
    if os.getenv("TOS_EMAIL_ENABLED", "true").lower() != "true":
        logger.info("✉️ TOS email disabled (TOS_EMAIL_ENABLED=false) — skipping.")
        return False
    try:
        subject = "Your Accepted Terms: EAiSER & MomntumAi"
        
        frontend_url = os.getenv("FRONTEND_URL", "https://www.eaiser.ai")
        if frontend_url.endswith("/"): frontend_url = frontend_url[:-1]
        
        # Path to the final Terms PDF
        tos_pdf = r"c:/Users/chris/OneDrive/Desktop/MomntumAi/momentum-frontend/public/TERMSand.pdf"
        
        display_name = name or 'User'
        inner_html = (
            '<h1 style="' + EMAIL_STYLES["h1"] + '">Terms of Service Confirmation</h1>'
            '<p style="' + EMAIL_STYLES["p"] + '">Hi ' + str(display_name) + ',</p>'
            '<div style="' + EMAIL_STYLES["box"] + 'font-weight:700;color:#065f46;">&#10004; Terms of Service Accepted</div>'
            '<p style="' + EMAIL_STYLES["p"] + '">Thank you for using EAiSER AI. This email confirms that you have reviewed and accepted our Terms of Service and Privacy Policy.</p>'
            '<p style="' + EMAIL_STYLES["p"] + '">As requested, we have attached a copy of these legal documents for your records. We\'re thrilled to have you on board. Together, we\'re building smarter, safer communities.</p>'
            '<p style="' + EMAIL_STYLES["p"] + 'font-weight:700;color:#0f172a;">&mdash; The EAiSER &amp; MomntumAi Team</p>'
        )
        html_content = build_branded_email(
            "Terms of Service Confirmation",
            inner_html,
            preheader="Your accepted Terms of Service — copy attached for your records.",
        )
        text_content = f"Hi {name or 'User'},\n\nThank you for using EAiSER.Ai. This email serves as confirmation that you have agreed to our Terms of Service. We have attached a copy for your records.\n\nThanks,\nThe EAiSER & MomntumAi Team"

        attachments = [tos_pdf] if os.path.exists(tos_pdf) else None
        return await send_email(email, subject, html_content, text_content, attachments=attachments)
    except Exception as e:
        logger.error(f"Failed to send TOS email to {email}: {e}")
        return False

# --------------------------------------------------------------------
# ✅ Government Portal Welcome Email
# --------------------------------------------------------------------

GOV_PORTAL_URL = "https://gov.eaiser.ai/login"

async def send_gov_welcome_email(
    email: str,
    name: str,
    department: str,
    city: str,
    zip_code: str,
    temporary_password: str
) -> bool:
    """
    Sends a professional welcome email to newly created Government Official accounts.
    """
    try:
        subject = f"Security Clearance Granted — EAiSER Government Portal ({department})"

        # ----------------------------
        # HTML EMAIL (branded shell)
        # ----------------------------
        credential_card = (
            '<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:16px;padding:24px 26px;margin:24px 0;">'
            '<div style="margin-bottom:18px;">'
            '<span style="font-size:11px;color:#94a3b8;display:block;margin-bottom:4px;text-transform:uppercase;letter-spacing:1px;font-weight:700;">Portal ID / Email</span>'
            '<span style="font-size:15px;color:#111;font-weight:700;">' + email.lower() + '</span>'
            '</div>'
            '<div>'
            '<span style="font-size:11px;color:#94a3b8;display:block;margin-bottom:4px;text-transform:uppercase;letter-spacing:1px;font-weight:700;">Temporary Security Password</span>'
            '<div style="background:#ffffff;border:1px dashed #cbd5e1;padding:12px 18px;border-radius:10px;font-family:Consolas,monospace;font-size:17px;color:#000;font-weight:800;display:table;margin-top:8px;">' + str(temporary_password) + '</div>'
            '</div>'
            '</div>'
        )
        scope_box = (
            '<div style="background:#f0fdf4;border:1px solid #dcfce7;padding:18px 20px;border-radius:12px;margin:24px 0;">'
            '<h4 style="margin:0 0 6px;color:#166534;font-size:14px;font-weight:800;">&#128640; Operation Scope</h4>'
            '<p style="margin:0;font-size:13px;color:#15803d;font-weight:500;line-height:1.6;">You are authorized to monitor and manage reports for the <strong>' + str(department) + '</strong> within ZIP context <strong>' + str(zip_code) + '</strong>. This dashboard is strictly for official use.</p>'
            '</div>'
        )
        inner_html = (
            '<h1 style="' + EMAIL_STYLES["h1"] + '">Official Clearance Granted</h1>'
            '<div style="display:inline-block;padding:5px 14px;background:#fffbeb;border:1px solid #fde68a;color:#b45309;border-radius:50px;font-size:11px;font-weight:800;margin-bottom:18px;text-transform:uppercase;">' + str(department) + ' &mdash; ' + str(city) + '</div>'
            '<p style="' + EMAIL_STYLES["p"] + '">Hello <strong>' + str(name) + '</strong>,</p>'
            '<p style="' + EMAIL_STYLES["p"] + '">Your official credentials for the EAiSER Government Portal have been provisioned by the EAiSER Global Operations team. You now have access to real-time citizen report routing and AI-driven department insights.</p>'
            + credential_card
            + scope_box
            + '<div style="' + EMAIL_STYLES["btn_wrap"] + '">'
            '<a href="' + GOV_PORTAL_URL + '" style="' + EMAIL_STYLES["btn"] + '">Access Gov Dashboard &rarr;</a>'
            '</div>'
        )
        html_content = build_branded_email(
            "Government portal access",
            inner_html,
            preheader="Your EAiSER Government Portal credentials are ready.",
        )
        text_content = f"""
Official Clearance Granted — EAiSER Gov

Hello {name},
Your credentials for the EAiSER Government Portal are ready.

Department: {department}
ZIP Context: {zip_code}

Portal ID: {email.lower()}
Temporary Password: {temporary_password}

Access the dashboard at: {GOV_PORTAL_URL}
"""
        return await send_email(email.lower(), subject, html_content, text_content)
    except Exception as e:
        logger.error(f"Failed to send gov welcome email: {e}")
        return False
