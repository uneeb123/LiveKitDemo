"""Public, signature-verified WhatsApp inbound call bridge. Run one worker."""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import sqlite3
import time
from contextlib import asynccontextmanager, closing
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from livekit import api

logger = logging.getLogger(__name__)


def create_app(*, config=None, connector=None):
    load_dotenv('.env.whatsapp')
    cfg = dict(os.environ) if config is None else config
    required = ('META_APP_SECRET', 'WHATSAPP_VERIFY_TOKEN', 'WHATSAPP_ACCESS_TOKEN',
                'WHATSAPP_PHONE_NUMBER_ID', 'WHATSAPP_WABA_ID')
    if any(not cfg.get(key) for key in required):
        raise ValueError('WhatsApp credentials and account IDs are required')
    db_path = Path(cfg.get('WHATSAPP_EVENT_DB', './data/whatsapp.sqlite3'))
    db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with closing(sqlite3.connect(db_path)) as db, db:
        db.execute('CREATE TABLE IF NOT EXISTS calls (id TEXT PRIMARY KEY, state TEXT, updated REAL)')
    db_path.chmod(0o600)
    lock = asyncio.Lock()

    def state(call_id, new_state=None):
        with closing(sqlite3.connect(db_path)) as db, db:
            if new_state:
                db.execute('INSERT INTO calls VALUES (?, ?, ?) ON CONFLICT(id) DO UPDATE '
                           'SET state=excluded.state, updated=excluded.updated',
                           (call_id, new_state, time.time()))
                db.execute('DELETE FROM calls WHERE updated < ?', (time.time() - 7 * 86400,))
            row = db.execute('SELECT state FROM calls WHERE id=?', (call_id,)).fetchone()
            return row[0] if row else None

    @asynccontextmanager
    async def lifespan(app):
        client = None
        if connector is None:
            client = api.LiveKitAPI()
            app.state.connector = client.connector
        else:
            app.state.connector = connector
        yield
        if client:
            await client.aclose()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.get('/health')
    async def health():
        return {'status': 'ok'}

    @app.get('/webhooks/whatsapp')
    async def verify(request: Request):
        q = request.query_params
        if q.get('hub.mode') != 'subscribe' or not hmac.compare_digest(
            q.get('hub.verify_token', ''), cfg['WHATSAPP_VERIFY_TOKEN']
        ):
            raise HTTPException(403, 'Verification failed')
        return PlainTextResponse(q.get('hub.challenge', ''))

    async def process(call):
        event, call_id = call.get('event'), call.get('id')
        if event not in ('connect', 'terminate') or not isinstance(call_id, str) or not call_id:
            return
        # Ignore stale connect retries; termination must still clean up an active call.
        if event == 'connect':
            try:
                age = time.time() - int(call['timestamp'])
            except (KeyError, ValueError, TypeError):
                raise HTTPException(400, 'Missing call timestamp') from None
            if age > 300 or age < -60:
                return
        async with lock:
            previous = await asyncio.to_thread(state, call_id)
            if previous == 'terminated' or (event == 'connect' and previous == 'accepted'):
                return
            bridge = app.state.connector
            try:
                async with asyncio.timeout(15):
                    if event == 'connect':
                        session = call.get('session', {})
                        caller = call.get('from')
                        if (call.get('direction') != 'USER_INITIATED' or
                                session.get('sdp_type') != 'offer'):
                            return
                        if not isinstance(caller, str) or not caller or len(caller) > 150:
                            raise HTTPException(400, 'Invalid caller')
                        if not isinstance(session.get('sdp'), str) or not session['sdp']:
                            raise HTTPException(400, 'Missing SDP')
                        await bridge.accept_whatsapp_call(api.AcceptWhatsAppCallRequest(
                            whatsapp_phone_number_id=cfg['WHATSAPP_PHONE_NUMBER_ID'],
                            whatsapp_api_key=cfg['WHATSAPP_ACCESS_TOKEN'],
                            whatsapp_cloud_api_version='26.0', whatsapp_call_id=call_id,
                            sdp={'type': 'offer', 'sdp': session['sdp']},
                            room_name='max-wa-' + hashlib.sha256(call_id.encode()).hexdigest()[:24],
                            participant_identity='whatsapp:' + caller,
                            agents=[api.RoomAgentDispatch(agent_name='max')],
                            wait_until_answered=False,
                        ))
                    else:
                        await bridge.disconnect_whatsapp_call(api.DisconnectWhatsAppCallRequest(
                            whatsapp_call_id=call_id,
                            disconnect_reason=api.DisconnectWhatsAppCallRequest.USER_INITIATED,
                        ))
            except api.TwirpError as exc:
                # A retry after a lost response must not create another session.
                if not ((event == 'connect' and exc.code == 'already_exists') or
                        (event == 'terminate' and exc.code == 'not_found')):
                    logger.warning('WhatsApp bridge rejected event (%s)', exc.code)
                    raise HTTPException(503, 'Call bridge unavailable') from None
            except TimeoutError:
                raise HTTPException(503, 'Call bridge timeout') from None
            await asyncio.to_thread(state, call_id, 'accepted' if event == 'connect' else 'terminated')

    @app.post('/webhooks/whatsapp')
    async def webhook(request: Request):
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 256_000:
                raise HTTPException(413, 'Payload too large')
        expected = 'sha256=' + hmac.new(cfg['META_APP_SECRET'].encode(), body, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(request.headers.get('x-hub-signature-256', ''), expected):
            raise HTTPException(401, 'Invalid signature')
        try:
            payload = json.loads(body)
            if payload.get('object') != 'whatsapp_business_account':
                return {'status': 'ignored'}
            for entry in payload.get('entry', []):
                if entry.get('id') != cfg['WHATSAPP_WABA_ID']:
                    continue
                for change in entry.get('changes', []):
                    value = change.get('value', {})
                    if (change.get('field') != 'calls' or
                            value.get('metadata', {}).get('phone_number_id') != cfg['WHATSAPP_PHONE_NUMBER_ID']):
                        continue
                    for call in value.get('calls', []):
                        await process(call)
        except (ValueError, TypeError, AttributeError):
            raise HTTPException(400, 'Invalid payload') from None
        return {'status': 'ok'}

    return app
