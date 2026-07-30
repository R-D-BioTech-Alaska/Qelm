import os
import sys

import Qelm2


def test_python_command_for_source(monkeypatch):
    monkeypatch.delattr(sys, 'frozen', raising=False)
    command = Qelm2._qelm_python_command('--qelm_prep_tokens')
    assert command[0] == sys.executable
    assert os.path.basename(command[1]) == 'Qelm2.py'
    assert command[2] == '--qelm_prep_tokens'


def test_python_command_for_executable(monkeypatch):
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    command = Qelm2._qelm_python_command('--qelm_prep_tokens')
    assert command == [sys.executable, '--qelm_prep_tokens']
