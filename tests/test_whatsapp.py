import hashlib
import hmac
import json
import time
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient
from livekit import api

from max_agent.whatsapp_service import create_app


def setup(tmp_path):
    connector = AsyncMock()
    cfg = {'META_APP_SECRET': 'test-secret', 'WHATSAPP_VERIFY_TOKEN': 'verify',
           'WHATSAPP_ACCESS_TOKEN': 'test-token', 'WHATSAPP_PHONE_NUMBER_ID': '123',
           'WHATSAPP_WABA_ID': '456', 'WHATSAPP_EVENT_DB': str(tmp_path / 'events.sqlite3')}
    return create_app(config=cfg, connector=connector), connector


def payload(event='connect'):
    return {'object': 'whatsapp_business_account', 'entry': [{'id': '456', 'changes': [
        {'field': 'calls', 'value': {'metadata': {'phone_number_id': '123'}, 'calls': [
            {'id': 'call-test', 'event': event, 'timestamp': str(int(time.time())),
             'direction': 'USER_INITIATED', 'from': '15550001111',
             'session': {'sdp_type': 'offer', 'sdp': 'v=0\r\n'}}]}}]}]}


def post(client, data, signed=True):
    body = json.dumps(data).encode()
    sig = 'sha256=' + hmac.new(b'test-secret', body, hashlib.sha256).hexdigest()
    return client.post('/webhooks/whatsapp', content=body,
                       headers={'x-hub-signature-256': sig if signed else 'bad'})


def test_verification_signature_and_account_scope(tmp_path):
    app, bridge = setup(tmp_path)
    with TestClient(app) as c:
        assert c.get('/webhooks/whatsapp', params={'hub.mode': 'subscribe',
                     'hub.verify_token': 'verify', 'hub.challenge': 'hello'}).text == 'hello'
        assert c.get('/webhooks/whatsapp').status_code == 403
        assert post(c, payload(), signed=False).status_code == 401
        wrong = payload()
        wrong['entry'][0]['id'] = 'other-account'
        assert post(c, wrong).status_code == 200
        bridge.accept_whatsapp_call.assert_not_called()


def test_call_lifecycle_and_durable_deduplication(tmp_path):
    app, bridge = setup(tmp_path)
    with TestClient(app) as c:
        assert post(c, payload()).status_code == 200
        req = bridge.accept_whatsapp_call.call_args.args[0]
        assert req.agents[0].agent_name == 'max'
        assert req.participant_identity == 'whatsapp:15550001111'
        assert req.sdp.sdp == 'v=0\r\n'
        assert post(c, payload()).status_code == 200
        assert bridge.accept_whatsapp_call.call_count == 1
    app2, bridge2 = setup(tmp_path)
    with TestClient(app2) as c:
        assert post(c, payload()).status_code == 200
        bridge2.accept_whatsapp_call.assert_not_called()
        assert post(c, payload('terminate')).status_code == 200
        assert post(c, payload('terminate')).status_code == 200
        assert post(c, payload()).status_code == 200
        assert bridge2.disconnect_whatsapp_call.call_count == 1
        bridge2.accept_whatsapp_call.assert_not_called()


def test_failed_accept_is_retryable_and_stale_calls_are_ignored(tmp_path):
    app, bridge = setup(tmp_path)
    bridge.accept_whatsapp_call.side_effect = api.TwirpError('unavailable', 'test', status=503)
    with TestClient(app) as c:
        assert post(c, payload()).status_code == 503
        bridge.accept_whatsapp_call.side_effect = None
        assert post(c, payload()).status_code == 200
        assert bridge.accept_whatsapp_call.call_count == 2
        stale = payload()
        call = stale['entry'][0]['changes'][0]['value']['calls'][0]
        call['id'] = 'old'; call['timestamp'] = '1'
        assert post(c, stale).status_code == 200
        assert bridge.accept_whatsapp_call.call_count == 2
