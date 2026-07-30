import multiprocessing
import os
import sys
import traceback


def _app_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _prepare_runtime():
    root = _app_dir()
    os.chdir(root)
    os.environ.setdefault('QELM_LAZY_NATIVE_IMPORTS', '1')
    os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
    os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
    os.environ.setdefault('NLTK_DATA', os.path.join(root, 'nltk_data'))
    os.environ.setdefault('MPLCONFIGDIR', os.path.join(root, '.qelm_cache', 'matplotlib'))

    if sys.stdout is None:
        sys.stdout = open(os.devnull, 'w', encoding='utf-8')
    if sys.stderr is None:
        sys.stderr = open(os.devnull, 'w', encoding='utf-8')

    graphviz_bin = os.path.join(root, 'graphviz', 'bin')
    if os.path.isdir(graphviz_bin):
        os.environ['PATH'] = graphviz_bin + os.pathsep + os.environ.get('PATH', '')


def _write_startup_error():
    path = os.path.join(_app_dir(), 'qelm_startup_error.txt')
    text = traceback.format_exc()
    try:
        with open(path, 'w', encoding='utf-8') as handle:
            handle.write(text)
    except Exception:
        pass
    try:
        from tkinter import messagebox
        messagebox.showerror('QELM', 'QELM could not start. Details were saved to qelm_startup_error.txt.')
    except Exception:
        pass


def main():
    _prepare_runtime()
    smoke_test = '--qelm_exe_smoke' in sys.argv
    if smoke_test:
        sys.argv.remove('--qelm_exe_smoke')
    try:
        import Qelm2
        Qelm2._install_crash_logger()
        if smoke_test:
            root = Qelm2.tk.Tk()
            Qelm2.QELM_GUI(root)
            root.update_idletasks()
            root.destroy()
            with open(os.path.join(_app_dir(), 'qelm_exe_smoke.ok'), 'w', encoding='utf-8') as handle:
                handle.write('ok')
            return
        Qelm2.main()
    except SystemExit:
        raise
    except Exception:
        _write_startup_error()
        raise


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
