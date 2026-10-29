"""Local ONVIF control: short, bounded moves; no autonomous patrol."""
import asyncio
import base64
import hashlib
import os
from datetime import datetime, timezone
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape
import httpx

PTZ='http://www.onvif.org/ver20/ptz/wsdl'
MEDIA='http://www.onvif.org/ver10/media/wsdl'
SCHEMA='http://www.onvif.org/ver10/schema'


class Controller:
    def __init__(self, config):
        self.config=config
        self.tokens={}
        self.locks={}
        self.generations={}

    async def call(self,camera,action,body,namespace=PTZ):
        account=self.config['onvif']
        nonce=os.urandom(16)
        created=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        digest=base64.b64encode(hashlib.sha1(nonce+created.encode()+account['password'].encode()).digest()).decode()
        header=f'''<Security s:mustUnderstand="1" xmlns="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd"><UsernameToken><Username>{escape(account['username'])}</Username><Password Type="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordDigest">{digest}</Password><Nonce EncodingType="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-soap-message-security-1.0#Base64Binary">{base64.b64encode(nonce).decode()}</Nonce><Created>{created}</Created></UsernameToken></Security>'''
        envelope=f'<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"><s:Header>{header}</s:Header><s:Body><{action} xmlns="{namespace}">{body}</{action}></s:Body></s:Envelope>'
        async with httpx.AsyncClient(timeout=5) as client:
            response=await client.post(camera['onvif_url'],content=envelope.encode(),headers={'Content-Type':f'application/soap+xml; charset=utf-8; action="{namespace}/{action}"'})
            response.raise_for_status()
        tree=ET.fromstring(response.content)
        if any(e.tag.endswith('Fault') for e in tree.iter()):
            raise ValueError('Camera rejected the request')
        return tree

    async def token(self,camera):
        key=camera['id']
        if key not in self.tokens:
            response=await self.call(camera,'GetProfiles','',MEDIA)
            token=next((e.attrib['token'] for e in response.iter() if e.tag.endswith('Profiles') and 'token' in e.attrib),None)
            if not token: raise ValueError('No camera profile')
            self.tokens[key]=token
        return escape(self.tokens[key])

    async def position(self,camera):
        token=await self.token(camera)
        response=await self.call(camera,'GetStatus',f'<ProfileToken>{token}</ProfileToken>')
        position=next((e for e in response.iter() if e.tag.endswith('PanTilt') and 'x' in e.attrib),None)
        if position is None: raise ValueError('Position unavailable')
        status=next((e.text for e in response.iter() if e.tag.endswith('PanTilt') and not e.attrib),None)
        return {'pan':float(position.attrib['x']),'tilt':float(position.attrib['y']),'moving':status=='MOVING'}

    async def move(self,camera,action,position=None):
        key=camera['id']
        # Stop bypasses the move queue. A slow position/move request must not
        # prevent a second owner request from stopping an already moving camera.
        # This follows tolpo's stop-without-capability-read control principle.
        if action=='stop':
            # Invalidate both queued work and work awaiting a profile, position,
            # or preset before yielding. Otherwise an old GetStatus response can
            # cause an AbsoluteMove to be sent AFTER this Stop completes.
            self.generations[key]=self.generations.get(key,0)+1
            token=await self.token(camera)
            await self.call(camera,'Stop',f'<ProfileToken>{token}</ProfileToken><PanTilt>true</PanTilt><Zoom>true</Zoom>')
            return {'ok':True}
        generation=self.generations.get(key,0)
        def cancelled():
            return generation!=self.generations.get(key,0)
        lock=self.locks.setdefault(key,asyncio.Lock())
        async with lock:
            if cancelled():return {'ok':True,'cancelled':True}
            token=await self.token(camera)
            if cancelled():return {'ok':True,'cancelled':True}
            prefix=f'<ProfileToken>{token}</ProfileToken>'
            if action=='home':
                presets=await self.call(camera,'GetPresets',prefix)
                if cancelled():return {'ok':True,'cancelled':True}
                preset=next((e.attrib['token'] for e in presets.iter() if e.tag.endswith('Preset') and 'token' in e.attrib),None)
                if preset:
                    await self.call(camera,'GotoPreset',prefix+f'<PresetToken>{escape(preset)}</PresetToken>')
                else:
                    await self.call(camera,'GotoHomePosition',prefix)
            else:
                if position is None:
                    position=await self.position(camera)
                    dx,dy={'left':(-.035,0),'right':(.035,0),'up':(0,.035),'down':(0,-.035)}[action]
                    position={'pan':max(-1,min(1,position['pan']+dx)),'tilt':max(-1,min(1,position['tilt']+dy))}
                if cancelled():return {'ok':True,'cancelled':True}
                await self.call(camera,'AbsoluteMove',prefix+f'<Position><PanTilt xmlns="{SCHEMA}" x="{position["pan"]}" y="{position["tilt"]}"/></Position>')
            return {'ok':True}
