"""Stop ordering with entirely mocked ONVIF calls; no camera is contacted."""
import asyncio
from unittest.mock import AsyncMock
from xml.etree import ElementTree as ET

import pytest

from backend.ptz import Controller


CAMERA={'id':'example'}
STATUS='<GetStatusResponse><Position><PanTilt x="0" y="0"/></Position></GetStatusResponse>'


def test_stop_invalidates_delayed_status_and_moves_already_in_the_queue():
    async def scenario():
        controller=Controller({});controller.tokens['example']='profile'
        entered,release=asyncio.Event(),asyncio.Event()
        async def call(camera,action,body,namespace=None):
            if action=='GetStatus':
                entered.set();await release.wait()
                return ET.fromstring(STATUS)
            return ET.fromstring('<OK/>')
        controller.call=AsyncMock(side_effect=call)
        active=asyncio.create_task(controller.move(CAMERA,'right'))
        await entered.wait()
        queued=asyncio.create_task(controller.move(CAMERA,'anchor',{'pan':.2,'tilt':.3}))
        await asyncio.sleep(0)  # Let the old anchor join the locked move queue.
        assert await asyncio.wait_for(controller.move(CAMERA,'stop'),.5)=={'ok':True}
        assert [c.args[1] for c in controller.call.await_args_list]==['GetStatus','Stop']
        release.set()
        assert await active=={'ok':True,'cancelled':True}
        assert await queued=={'ok':True,'cancelled':True}
        assert [c.args[1] for c in controller.call.await_args_list]==['GetStatus','Stop']
        # A deliberate move requested after Stop belongs to the new generation.
        assert await controller.move(CAMERA,'left')=={'ok':True}
        assert [c.args[1] for c in controller.call.await_args_list][-2:]==['GetStatus','AbsoluteMove']
    asyncio.run(scenario())


@pytest.mark.parametrize('presets',['<GetPresetsResponse/>','<GetPresetsResponse><Preset token="perch"/></GetPresetsResponse>'])
def test_stop_invalidates_delayed_home_discovery(presets):
    async def scenario():
        controller=Controller({});controller.tokens['example']='profile'
        entered,release=asyncio.Event(),asyncio.Event()
        async def call(camera,action,body,namespace=None):
            if action=='GetPresets':
                entered.set();await release.wait()
                return ET.fromstring(presets)
            return ET.fromstring('<OK/>')
        controller.call=AsyncMock(side_effect=call)
        home=asyncio.create_task(controller.move(CAMERA,'home'))
        await entered.wait()
        await asyncio.wait_for(controller.move(CAMERA,'stop'),.5)
        release.set()
        assert await home=={'ok':True,'cancelled':True}
        assert [c.args[1] for c in controller.call.await_args_list]==['GetPresets','Stop']
    asyncio.run(scenario())


def test_stop_invalidates_move_waiting_for_its_first_profile():
    async def scenario():
        controller=Controller({})
        entered,release=asyncio.Event(),asyncio.Event()
        profiles=0
        async def call(camera,action,body,namespace=None):
            nonlocal profiles
            if action=='GetProfiles':
                profiles+=1
                if profiles==1:entered.set();await release.wait()
                return ET.fromstring('<GetProfilesResponse><Profiles token="profile"/></GetProfilesResponse>')
            return ET.fromstring('<OK/>')
        controller.call=AsyncMock(side_effect=call)
        old=asyncio.create_task(controller.move(CAMERA,'up'))
        await entered.wait()
        await asyncio.wait_for(controller.move(CAMERA,'stop'),.5)
        release.set()
        assert await old=={'ok':True,'cancelled':True}
        assert [c.args[1] for c in controller.call.await_args_list]==['GetProfiles','GetProfiles','Stop']
    asyncio.run(scenario())


def test_failed_stop_does_not_revive_previously_queued_work():
    async def scenario():
        controller=Controller({});controller.tokens['example']='profile'
        entered,release=asyncio.Event(),asyncio.Event()
        async def call(camera,action,body,namespace=None):
            if action=='GetStatus':
                entered.set();await release.wait()
                return ET.fromstring(STATUS)
            if action=='Stop':raise ValueError('Fictional rejected stop')
            return ET.fromstring('<OK/>')
        controller.call=AsyncMock(side_effect=call)
        old=asyncio.create_task(controller.move(CAMERA,'down'))
        await entered.wait()
        with pytest.raises(ValueError,match='Fictional rejected stop'):
            await controller.move(CAMERA,'stop')
        release.set()
        assert await old=={'ok':True,'cancelled':True}
        assert [c.args[1] for c in controller.call.await_args_list]==['GetStatus','Stop']
    asyncio.run(scenario())


def test_stop_generation_is_isolated_to_its_camera():
    async def scenario():
        controller=Controller({});controller.tokens={'example':'profile','other':'profile'}
        entered,release=asyncio.Event(),asyncio.Event()
        async def call(camera,action,body,namespace=None):
            if action=='GetStatus':
                entered.set();await release.wait()
                return ET.fromstring(STATUS)
            return ET.fromstring('<OK/>')
        controller.call=AsyncMock(side_effect=call)
        other=asyncio.create_task(controller.move({'id':'other'},'right'))
        await entered.wait()
        await controller.move(CAMERA,'stop')
        release.set()
        assert await other=={'ok':True}
        assert [(c.args[0]['id'],c.args[1]) for c in controller.call.await_args_list]==[
            ('other','GetStatus'),('example','Stop'),('other','AbsoluteMove')]
    asyncio.run(scenario())
