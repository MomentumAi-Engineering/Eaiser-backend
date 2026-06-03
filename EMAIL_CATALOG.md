# EAiSER — Email Catalog & Cadence

Complete inventory of every email the backend sends, with trigger/cadence, recipients, subject, and body summary.

## Global config
- **Provider:** Postmark API (`https://api.postmarkapp.com/email`)
- **FROM:** `alert@momntumai.com` (env `EMAIL_USER`)
- **Reply-To (citizen/authority threads):** `reports@inbound.eaiser.ai` (env `POSTMARK_INBOUND_EMAIL`) — replies route back through the inbound webhook
- **Dry-run:** `EMAIL_DRY_RUN=true` skips all sends
- **Retry:** one automatic retry on failure (except 401)
- **Cadence:** Everything is **event-driven**. There are **no scheduled/recurring/digest emails** (no Celery `beat_schedule`).

---

## A. Reporting & routing (citizen ↔ authority)

### 1. Authority Smart Alert — `send_formatted_ai_alert()`
- **File:** `app/services/email_service.py`
- **Trigger/cadence:** Immediately (background) when a report is submitted / auto-submitted / approved.
- **To:** primary authority · **Cc:** all other authorities (one shared thread) · **Reply-To:** inbound.
- **Subject:** `EAiSER Alert – {issue_type} (Priority: {priority}, ID: {report_id})`
- **Body:** "EAiSER SMART ALERT" header → **Departments Notified** list (collaboration note) → AI report body (or rebuilt fallback if missing) → "Official Communication: reply to coordinate with the citizen" → footer.

### 2. Authority Standard Alert — `send_authority_email()`
- **File:** `app/routes/issues.py`
- **Trigger/cadence:** Immediately on the approval/route flow.
- **To:** primary authority · **Cc:** all other authorities · **Reply-To:** inbound.
- **Subject:** `[ID: {oid}] CIVIC ALERT: {issue_type}` (or `[ID: {oid}] Flagged for Review: {issue_type}` when AI detected no issue)
- **Body:** "Incident Report Standard" w/ priority badge → §1 Overview (summary, type, date, location, map, GPS, ID) → §2 Photographic Evidence → §4 Incident Routing (all depts, "in the spirit of collaboration") → footer. Text version is a structured "Action Required" list.

### 3. Reporter Copy (citizen confirmation) — `_send_reporter_copy()`
- **File:** `app/routes/issues.py`
- **Trigger/cadence:** Once, immediately after submission (only if reporter provided an email). **Best-effort.**
- **To:** reporter only · **No Cc** (authorities NOT copied).
- **Subject:** `Your EAiSER report ({issue_id}) was submitted`
- **Body:** Thanks + Issue/Location/Reference + "create a free account to track" + **"Authorities notified:" list** (or "being reviewed by the EAiSER team" when none).
- ⚠️ Only wired into legacy `issues.py` create flow — **not** the optimized v2 auto-submit flow.

### 4 & 5. Report Status Update (Approved / Declined) — `notify_user_status_change()`
- **File:** `app/services/email_service.py` · triggered from `app/routes/admin_review.py` (`/approve`, `/decline`).
- **Trigger/cadence:** Immediately when an admin approves or declines.
- **To:** reporter (`reporter_email`/`user_email`) · No Cc.
- **Subject:** `Your EAiSER report – {issue_type} - Issue ID: {issue_id}.` (declined: no trailing period)
- **Body (approved):** "we have successfully provided the details to **{authority names}** for further action." + Back to Dashboard CTA.
- **Body (declined):** "we have not identified any issues that require notification to authorities."

### 6. Authority Reply → Citizen — `email_webhook.py`
- **Trigger/cadence:** Immediately when an authority replies to the thread.
- **To:** citizen · **Reply-To:** inbound. Forwards attachments.
- **Subject:** `Official Response: Report #{issue_id}`
- **Body:** "An official representative has reviewed your report…" + cleaned reply text + "reply to continue the conversation."

### 7. Citizen Reply → Authorities — `email_webhook.py`
- **Trigger/cadence:** Immediately when the citizen replies. Sent individually to each authority (loop, not Cc).
- **To:** each authority · **Reply-To:** inbound.
- **Subject:** `NEW FOLLOW-UP: Incident #{issue_id}`
- **Body:** "Citizen Follow-up Received" + cleaned message + operational guidance.

---

## B. Account & auth (citizen app users) — `app/services/email_service.py`

### 8. Email Verification — `send_verification_email()`
- **Trigger:** New user signup. **To:** user.
- **Subject:** `Verify your EAiSER AI Account`
- **Body:** "Verify My Account" CTA → `{FRONTEND_URL}/verify-email?token=…`. **Expires 24h.**

### 9. User Welcome — `send_user_welcome_email()`
- **Trigger:** Account creation / after verification. **To:** user.
- **Subject:** `Welcome to EAiSER AI – Your Journey to a Smarter Community Begins 🚀`
- **Body:** Feature highlights (Impact via Imagery / Autonomous Routing / Real-time Visibility) + "Initialize My Dashboard" CTA.

### 10. Password Reset — `send_password_reset_email()`
- **Trigger:** "Forgot password". **To:** user.
- **Subject:** `Reset Your EAiSER Password`
- **Body:** "Reset My Password" CTA → `{FRONTEND_URL}/reset-password?token=…`. **Expires 15 min.**

### 11. Terms of Service Confirmation — `send_tos_email()`
- **Trigger:** User accepts TOS. **To:** user. **Attachment:** TOS PDF. **Kill-switch:** `TOS_EMAIL_ENABLED=false`.
- **Subject:** `Your Accepted Terms: EAiSER & MomntumAi`
- **Body:** "Terms of Service Accepted" + attached copy.
- ⚠️ PDF path is a hardcoded Windows path (`c:/Users/chris/…/TERMSand.pdf`) — won't resolve on the server.

---

## C. Admin / internal

### 12. Admin Welcome — `send_admin_welcome_email()`
- **File:** `app/services/admin_email_service.py` (and a duplicate in `email_service.py`). Trigger: admin account created (`admin_review.py`).
- **To:** new admin.
- **Subject:** `Welcome to EAiSER — {role} Access Granted`
- **Body:** temp password + role permissions + "Launch Admin Console" → `https://www.eaiser.ai/admin`.

### 13. Admin 2FA Code — `security_service.py` `send_2fa_email()`
- **Trigger:** admin login 2FA. **To:** admin.
- **Subject:** `EAiSER Admin - Two-Factor Authentication Code`
- **Body:** 6-digit code, **expires 10 min**.

### 14. City Partnership Inquiry Alert — `app/routes/inquiry.py`
- **Trigger:** `/city-inquiry` form submit. **To:** `CITY_INQUIRY_EMAIL` (default `eaiser@momntumai.com`).
- **Subject:** `🏙️ New City Partnership Request: {city}`
- **Body:** name/role/email/city + message.

### 15. Sales Demo Request — `app/gov/billing.py` `_notify_sales_demo_request()`
- **Trigger:** `/gov/billing/demo-request` submit. **To:** `SALES_NOTIFY_EMAIL` (default `sales@eaiser.ai`).
- **Subject:** `[New Demo Request] {cityName}, {state} — {tier}`
- **Body:** all demo-request fields.

---

## D. Government portal (gov partners)

### 16 / 17. Gov Official Welcome (two implementations)
- `email_service.py` `send_gov_welcome_email()` → **Subject:** `Security Clearance Granted — EAiSER Government Portal ({department})`
- `app/gov/gov_auth.py` (`/gov/setup-account`) → **Subject:** `Welcome to EAiSER Government Portal - {city} {department}`
- **To:** official. **Body:** portal credentials (email + temp password), dept/ZIP scope, gov dashboard link. ⚠️ Two different templates for the same purpose.

### 18. Gov Password Reset — `app/gov/gov_auth.py` (`/gov/reset-password`)
- **To:** official. **Subject:** `EAiSER Portal — Emergency Access Reset: {city}` · Body: new temp password.

### 19. Team Invitation — `app/gov/onboarding.py`
- **Trigger:** onboarding completion w/ invites. **Subject:** `Invitation to join EAiSER Civic` · "Accept Invitation" link, **expires 7 days**.

### 20. Activation Link (Resend) — `app/gov/activation.py`
- **Subject:** `Your EAiSER activation link` · "Activate Account" link, **7 days**.

### 21. Team Member Invite (Activation) — `app/gov/activation.py`
- **Subject:** `You've been invited to EAiSER Civic ({role})` · "Accept Invitation" link, **7 days**.

### 22. Checkout / Org Activation — `app/gov/billing.py` `_send_activation_email()`
- **Trigger:** gov org completes checkout. **To:** buyer.
- **Subject:** `Welcome to EAiSER Civic — Activate your {city} account`
- **Body:** "Activate My Account" link (**7 days**), Plan {tier}, **30-day free trial**.

---

## Cross-cutting observations
1. **FROM is `alert@momntumai.com`** (MomntumAi domain), while links/branding are `eaiser.ai` — worth aligning for deliverability/trust.
2. **No recurring emails** — no digests, reminders, or follow-ups. All one-shot on events.
3. **Reporter copy** is only sent from the legacy `issues.py` flow, not the optimized v2 auto-submit path.
4. **Duplicate templates** for admin-welcome and gov-welcome (two implementations each) — risk of drift.
5. **TOS PDF path** is a hardcoded Windows path — broken on the server.
6. **Citizen→authority follow-ups** are sent per-authority in a loop (no shared Cc thread), unlike the initial alert which uses To+Cc.
