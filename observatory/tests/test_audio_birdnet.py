import numpy as np
import pytest

from backend.audio_birdnet import predict_windows


class FakeInterpreter:
    def get_input_details(self):return [{'shape':[1,144000],'dtype':np.float32,'index':0}]
    def get_output_details(self):return [{'shape':[1,3],'index':1}]
    def set_tensor(self,index,value):self.input=value
    def invoke(self):pass
    def get_tensor(self,index):return np.array([[-2,0,2]],dtype=np.float32)


def test_explicit_birdnet_contract_scores_and_padding():
    fake=FakeInterpreter()
    result=predict_windows(fake,['fictional-a','fictional-b','fictional-c'],np.ones(48000*4,np.float32),top_k=2)
    assert len(result)==2
    assert result[0]['top_scores'][0]=={'label':'fictional-c','score':.880797}
    assert result[1]['start_seconds']==3 and result[1]['end_seconds']==4
    assert result[1]['padded_seconds']==2
    assert np.all(fake.input[0,48000:]==0)
    assert result[0]['validated'] is False and result[0]['individual'] is None


def test_mismatched_labels_are_rejected_and_subsecond_tail_is_not_fabricated():
    with pytest.raises(ValueError,match='label count'):
        predict_windows(FakeInterpreter(),['one'],np.zeros(144000),top_k=1)
    assert predict_windows(FakeInterpreter(),['a','b','c'],np.zeros(24000),top_k=1)==[]
