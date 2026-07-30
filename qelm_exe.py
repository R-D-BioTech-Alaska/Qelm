import multiprocessing
import os
import sys
import traceback


def _app_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def _resource_dir():
    if getattr(sys, 'frozen', False):
        return os.path.abspath(getattr(sys, '_MEIPASS', _app_dir()))
    return os.path.dirname(os.path.abspath(__file__))


def _prepare_runtime():
    app_root = _app_dir()
    resource_root = _resource_dir()
    os.chdir(resource_root)

    os.environ.setdefault('QELM_LAZY_NATIVE_IMPORTS', '1')
    os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
    os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
    os.environ.setdefault('QELM_BUNDLE_ROOT', resource_root)
    os.environ.setdefault('NLTK_DATA', os.path.join(resource_root, 'nltk_data'))
    os.environ.setdefault('MPLCONFIGDIR', os.path.join(app_root, '.qelm_cache', 'matplotlib'))

    if sys.stdout is None:
        sys.stdout = open(os.devnull, 'w', encoding='utf-8')
    if sys.stderr is None:
        sys.stderr = open(os.devnull, 'w', encoding='utf-8')

    tcl_data = os.path.join(resource_root, '_tcl_data')
    tk_data = os.path.join(resource_root, '_tk_data')
    if os.path.isdir(tcl_data):
        os.environ['TCL_LIBRARY'] = tcl_data
    if os.path.isdir(tk_data):
        os.environ['TK_LIBRARY'] = tk_data

    graphviz_bin = os.path.join(resource_root, 'graphviz', 'bin')
    if os.path.isdir(graphviz_bin):
        os.environ['PATH'] = os.environ.get('PATH', '') + os.pathsep + graphviz_bin


def _output_path(name):
    override = os.environ.get('QELM_OUTPUT_DIR', '').strip()
    root = os.path.abspath(override) if override else _app_dir()
    try:
        os.makedirs(root, exist_ok=True)
    except Exception:
        root = os.path.expanduser('~')
    return os.path.join(root, name)


def _write_startup_error(show_dialog=True):
    path = _output_path('qelm_startup_error.txt')
    text = traceback.format_exc()
    try:
        with open(path, 'w', encoding='utf-8') as handle:
            handle.write(text)
    except Exception:
        pass
    if not show_dialog:
        return
    try:
        from tkinter import messagebox
        messagebox.showerror('QELM', f'QELM could not start. Details were saved to {path}.')
    except Exception:
        pass


def main():
    _prepare_runtime()
    smoke_test = '--qelm_exe_smoke' in sys.argv
    if smoke_test:
        sys.argv.remove('--qelm_exe_smoke')
    try:
        import Qelm2
        Qelm2._install_crash_logger(_output_path('qelm_crashlog.txt'))
        if smoke_test:
            root = Qelm2.tk.Tk()
            Qelm2.QELM_GUI(root)
            root.update_idletasks()
            root.destroy()
            with open(_output_path('qelm_exe_smoke.ok'), 'w', encoding='utf-8') as handle:
                handle.write('ok')
            return
        Qelm2.main()
    except SystemExit:
        raise
    except Exception:
        _write_startup_error(show_dialog=not smoke_test)
        raise


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
