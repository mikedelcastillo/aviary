"""HLS identity, restart, sliding-window and crash-recovery invariants."""
from datetime import datetime, timezone
import os
import pytest
from io import BytesIO
import threading

from backend.recorder import command_for
from backend.recorder_hls import HLSPublisher, read_playlist


def incoming(publisher, sequences, *, timestamp=1_800_000_000, end=False):
    lines=['#EXTM3U','#EXT-X-TARGETDURATION:2',f'#EXT-X-MEDIA-SEQUENCE:{sequences[0]}']
    for sequence in sequences:
        name=f's{sequence:010d}.ts'
        (publisher.folder/name).write_bytes(b'fictional complete TS')
        date=datetime.fromtimestamp(timestamp+(sequence-sequences[0])*2,timezone.utc).isoformat()
        lines.extend(['#EXTINF:2.000000,',f'#EXT-X-PROGRAM-DATE-TIME:{date}',name])
    if end:lines.append('#EXT-X-ENDLIST')
    temporary=publisher.incoming.with_suffix('.new')
    temporary.write_text('\n'.join(lines)+'\n')
    temporary.replace(publisher.incoming)


def test_restart_preserves_old_identity_pdt_and_marks_pts_reset(tmp_path):
    first=HLSPublisher(tmp_path/'live.m3u8')
    first.start=100;first.last=99
    incoming(first,[100,101,102]);assert first.publish()
    before,_=read_playlist(first.playlist)
    restarted=HLSPublisher(first.playlist)
    assert restarted.start==103
    assert not restarted.publish()  # Old incoming cannot become fresh output.
    incoming(restarted,[103,104],timestamp=1_800_000_010,end=True)
    assert restarted.publish()
    after,_=read_playlist(first.playlist)
    assert after[:3]==before
    assert [(s.sequence,s.discontinuity) for s in after]==[(100,0),(101,0),(102,0),(103,1),(104,1)]
    text=first.playlist.read_text()
    assert text.count('#EXT-X-DISCONTINUITY\n')==1
    assert '#EXT-X-ENDLIST' not in text


def test_discontinuity_number_survives_window_slide_and_process_restart(tmp_path):
    first=HLSPublisher(tmp_path/'live.m3u8',window=3)
    first.start=100;first.last=99
    incoming(first,[100,101,102]);first.publish()
    second=HLSPublisher(first.playlist,window=3)
    incoming(second,[103,104,105],timestamp=1_800_000_008);second.publish()
    assert '#EXT-X-DISCONTINUITY-SEQUENCE:1\n' in first.playlist.read_text()
    assert '#EXT-X-DISCONTINUITY\n' not in first.playlist.read_text()
    third=HLSPublisher(first.playlist,window=3)
    incoming(third,[106],timestamp=1_800_000_016);third.publish()
    segments,_=read_playlist(first.playlist)
    assert [(s.sequence,s.discontinuity) for s in segments]==[(104,1),(105,1),(106,2)]
    incoming(third,[106,107,108],timestamp=1_800_000_016);third.publish()
    assert '#EXT-X-DISCONTINUITY-SEQUENCE:2\n' in first.playlist.read_text()


def test_long_outage_drops_stale_history_but_keeps_removed_uris_available(tmp_path):
    first=HLSPublisher(tmp_path/'live.m3u8')
    first.start=100;first.last=99
    incoming(first,[100,101,102]);first.publish()
    old_file=tmp_path/'s0000000100.ts';os.utime(old_file,(1,1))
    restarted=HLSPublisher(first.playlist)
    incoming(restarted,[103],timestamp=1_800_000_100);restarted.publish()
    rows,_=read_playlist(first.playlist)
    assert [row.sequence for row in rows]==[103]
    assert rows[0].discontinuity==1
    assert old_file.exists() and old_file.stat().st_mtime>1
    os.utime(old_file,(1,1));restarted.last_cleanup=0;restarted.cleanup()
    assert not old_file.exists()
    assert (tmp_path/rows[0].uri).exists()


def test_incomplete_segment_does_not_replace_the_public_playlist(tmp_path):
    publisher=HLSPublisher(tmp_path/'live.m3u8')
    publisher.start=100;publisher.last=99
    incoming(publisher,[100]);publisher.publish();before=publisher.playlist.read_bytes()
    incoming(publisher,[100,101]);(tmp_path/'s0000000101.ts').unlink()
    assert not publisher.publish()
    assert publisher.playlist.read_bytes()==before


def test_published_uri_sequence_and_pdt_are_immutable_through_multiple_restarts(tmp_path):
    known={}
    for generation in range(4):
        publisher=HLSPublisher(tmp_path/'live.m3u8',window=5)
        for count in range(1,9):
            incoming(publisher,list(range(publisher.start,publisher.start+count)),timestamp=1_800_000_000+generation*20)
            publisher.publish()
            rows,_=read_playlist(publisher.playlist)
            assert len(rows)<=5
            for row in rows:
                assert known.setdefault(row.uri,row)==row


def test_recorder_uses_private_sequential_playlist_and_no_terminal_endlist(tmp_path):
    publisher=HLSPublisher(tmp_path/'hls/fixture/live.m3u8')
    command=command_for({'runtime':str(tmp_path),'relay_rtsp':'rtsp://localhost'}, {'id':'fixture'},'libx264',publisher=publisher)
    output=command[command.index('tee')+1]
    assert str(publisher.incoming) in output
    assert f'start_number={publisher.start}' in output
    assert 'hls_start_number_source=epoch' not in output
    assert 'omit_endlist' in output and 'append_list' not in output


def test_migration_from_old_ffmpeg_playlist_without_discontinuity_sequence(tmp_path):
    old=HLSPublisher(tmp_path/'live.m3u8')
    incoming(old,[100,101],end=True)
    old.incoming.replace(old.playlist)  # Exactly the old FFmpeg tag layout.
    migrated=HLSPublisher(old.playlist)
    incoming(migrated,[102],timestamp=1_800_000_006);migrated.publish()
    rows,_=read_playlist(old.playlist)
    assert [(row.sequence,row.discontinuity) for row in rows]==[(100,0),(101,0),(102,1)]
    assert '#EXT-X-ENDLIST' not in old.playlist.read_text()


def test_malformed_persisted_playlist_is_quarantined_without_uri_reuse(tmp_path):
    public=tmp_path/'live.m3u8';public.write_text('truncated invalid state')
    existing=tmp_path/'s999999999999999.ts';existing.write_bytes(b'previously advertised')
    publisher=HLSPublisher(public)
    assert (tmp_path/'live.m3u8.invalid').read_text()=='truncated invalid state'
    assert publisher.start==1000000000000000 and publisher.generation==1
    incoming(publisher,[publisher.start]);publisher.publish()
    assert existing.read_bytes()==b'previously advertised'
    assert len(read_playlist(public)[0])==1


def test_zeroed_legacy_manifest_uses_next_segment_id_without_epoch_jump(tmp_path):
    public=tmp_path/'live.m3u8';public.write_text('')
    for sequence in (1793247000,1793247001,1793247002):
        (tmp_path/f's{sequence}.ts').write_bytes(b'previous complete segment')
    publisher=HLSPublisher(public)
    assert publisher.start==1793247003 and publisher.generation==1


def test_corrupt_public_recovers_durable_high_id_and_discontinuity_checkpoint(tmp_path):
    publisher=HLSPublisher(tmp_path/'live.m3u8')
    publisher.start=1793247441657148;publisher.last=publisher.start-1;publisher.generation=20755
    incoming(publisher,[publisher.start,publisher.start+1]);publisher.publish()
    before,_=read_playlist(publisher.playlist)
    publisher.playlist.write_text('')
    recovered=HLSPublisher(publisher.playlist)
    assert recovered.start==before[-1].sequence+1
    assert recovered.generation==20756
    incoming(recovered,[recovered.start],timestamp=1_800_000_006);recovered.publish()
    after,_=read_playlist(publisher.playlist)
    assert after[:2]==before
    assert after[-1].sequence==before[-1].sequence+1 and after[-1].discontinuity==20756


def test_valid_public_high_ids_remain_primary_over_older_checkpoint(tmp_path):
    publisher=HLSPublisher(tmp_path/'live.m3u8')
    publisher.start=1793247441657148;publisher.last=publisher.start-1;publisher.generation=20755
    incoming(publisher,[publisher.start]);publisher.publish()
    checkpoint=publisher.checkpoint.read_text()
    incoming(publisher,[publisher.start,publisher.start+1]);publisher.publish()
    publisher.checkpoint.write_text(checkpoint)
    recovered=HLSPublisher(publisher.playlist)
    assert recovered.start==publisher.start+2 and recovered.generation==20756


def test_target_duration_stays_fixed_when_final_segment_is_short(tmp_path):
    publisher=HLSPublisher(tmp_path/'live.m3u8')
    publisher.start=100;publisher.last=99
    incoming(publisher,[100]);publisher.publish()
    incoming(publisher,[101],timestamp=1_800_000_002)
    publisher.incoming.write_text(publisher.incoming.read_text().replace('TARGETDURATION:2','TARGETDURATION:1').replace('EXTINF:2.000000','EXTINF:0.500000'))
    publisher.publish()
    assert read_playlist(publisher.playlist)[1]==2
    before=publisher.playlist.read_bytes()
    incoming(publisher,[102],timestamp=1_800_000_004)
    publisher.incoming.write_text(publisher.incoming.read_text().replace('EXTINF:2.000000','EXTINF:4.000000'))
    with pytest.raises(ValueError):publisher.publish()
    assert publisher.playlist.read_bytes()==before


def test_publication_failure_stops_child_before_record_returns(tmp_path,monkeypatch):
    from backend import recorder
    stop=threading.Event()
    class BrokenPublisher:
        def __init__(self,*_):pass
        def publish(self):raise ValueError('fictional corrupt incoming playlist')
    class Child:
        pid=123456
        stdout=BytesIO();stderr=BytesIO()
        returncode=None
        def poll(self):return self.returncode
        def wait(self):return self.returncode
    child=Child();stops=[]
    def terminate(process,*_,**__):
        assert process is child
        stops.append(True);child.returncode=255;stop.set()
    monkeypatch.setattr(recorder,'HLSPublisher',BrokenPublisher)
    monkeypatch.setattr(recorder,'command_for',lambda *_,**__:['fictional-encoder'])
    monkeypatch.setattr(recorder.subprocess,'Popen',lambda *_,**__:child)
    monkeypatch.setattr(recorder,'stop_child',terminate)
    recorder.record({'runtime':str(tmp_path),'video_encoder':'libx264'},{'id':'fixture'},stop)
    assert stops==[True] and child.returncode==255
