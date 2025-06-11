project('GUI Project', 'cpp',
  version : '0.1',
  default_options : [
    'warning_level=2',
    'cpp_std=c++20',
    'c_std=c17',
    'default_library=static'
  ]
)

if host_machine.system() == 'windows'
  sdl2_dep = subproject('sdl2_windows').get_variable('sdl2_windows_dep')
else
  sdl2_dep = dependency('sdl2')
endif

animationwindow_dep = dependency('animationwindow', fallback: ['animationwindow', 'animationwindow_dep'])
std_lib_facilities_dep = dependency('std_lib_facilities', fallback: ['std_lib_facilities', 'std_lib_facilities_dep'])

if host_machine.system() == 'linux'
  compiler_flags = ['-Wconversion', '-fdiagnostics-color=always', '-Werror=return-type']
else
  compiler_flags = ['-Wconversion', '-fdiagnostics-color=always', '-Werror=return-type', '-fcolor-diagnostics', '-fansi-escape-codes']
endif

src = files(
  'sourcecode/main.cpp',
  'sourcecode/dataPath.cpp',
  'sourcecode/fastNetwork/matrix2.cpp',
  'sourcecode/fastNetwork/networkFast.cpp',
  'sourcecode/slowNetwork/matrix.cpp',
  'sourcecode/slowNetwork/network.cpp',
  'sourcecode/MNIST/mnist_loader.cpp',
  'sourcecode/MNIST/visuelMnist.cpp',
  'sourcecode/xor/xor.cpp'
)

executable(
  'program',
  src,
  dependencies: [
    animationwindow_dep,
    sdl2_dep,
    std_lib_facilities_dep,
    dependency('threads')
  ],
  include_directories: include_directories('sourcecode'),  # 🔧 ← dette er løsningen
  cpp_args: compiler_flags,
  link_args: ['-Wl,-subsystem,console']
)
