from pathlib import Path

from PyInstaller.utils.hooks import collect_all, copy_metadata


root = Path(SPECPATH)
datas = [
    (str(root / 'QLM'), 'QLM'),
    (str(root / 'Datasets'), 'Datasets'),
    (str(root / 'docs' / 'images' / 'qelm_logo_small.png'), 'docs/images'),
    (str(root / 'build' / 'nltk_data'), 'nltk_data'),
    (str(root / 'build' / 'graphviz'), 'graphviz'),
    (str(root / 'README.md'), '.'),
    (str(root / 'LICENSE'), '.'),
    (str(root / 'requirements.txt'), '.'),
]
binaries = []
hiddenimports = [
    'Qelm2',
    'QelmInference',
    'QelmTokenizer',
    'QELMChatUI',
]

for package in ('qiskit', 'qiskit_aer', 'qiskit_ibm_runtime'):
    try:
        package_data, package_binaries, package_hidden = collect_all(package)
        datas += package_data
        binaries += package_binaries
        hiddenimports += package_hidden
    except Exception:
        pass

for package in (
    'qelm',
    'qiskit',
    'qiskit-aer',
    'qiskit-ibm-runtime',
    'numpy',
    'scipy',
    'nltk',
    'tensorflow',
    'keras',
    'psutil',
    'matplotlib',
    'pydot',
    'graphviz',
):
    try:
        datas += copy_metadata(package)
    except Exception:
        pass

analysis = Analysis(
    ['qelm_exe.py'],
    pathex=[str(root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'transformers', 'llama_cpp', 'gguf'],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(analysis.pure)

exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name='QELM',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=str(root / 'build' / 'qelm.ico'),
    contents_directory='.',
)

bundle = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='QELM',
)
