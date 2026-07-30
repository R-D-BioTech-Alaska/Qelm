import os
import sys

import Qelm2
import qelm_exe


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


def test_onefile_resource_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    monkeypatch.setattr(sys, '_MEIPASS', str(tmp_path), raising=False)
    assert qelm_exe._resource_dir() == os.path.abspath(str(tmp_path))


def test_frozen_app_directory_is_executable_parent(monkeypatch, tmp_path):
    executable = tmp_path / 'QELM.exe'
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    monkeypatch.setattr(sys, 'executable', str(executable))
    assert qelm_exe._app_dir() == os.path.abspath(str(tmp_path))
