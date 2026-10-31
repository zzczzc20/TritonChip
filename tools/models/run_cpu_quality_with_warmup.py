#!/usr/bin/env python3
"""Run existing full CPU quality checks after one dummy decode warmup."""
import sys
import chat
import chat_quality

if '--device' not in sys.argv or sys.argv[sys.argv.index('--device')+1]!='cpu':
    raise SystemExit('CPU diagnostic only; explicitly pass --device cpu')
original=chat.ChatApplication.generate

def generate(self,prompt,images=(),*args,**kwargs):
    if self.session is None:
        session=self.device_session()
        inputs,_=chat.make_inputs(self.spec,self.processor,[chat.user_message(prompt,images)])
        session.warmup_decode(inputs)
    return original(self,prompt,images,*args,**kwargs)

chat.ChatApplication.generate=generate
try:
    raise SystemExit(chat_quality.main())
finally:
    chat.ChatApplication.generate=original
