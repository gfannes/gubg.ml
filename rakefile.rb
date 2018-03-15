require_relative("../gubg.build/bootstrap.rb")
require("gubg/shared")
require("gubg/build/Executable")
require("gubg/build/Library")
include GUBG

task :default => :help
task :help do
    puts('declare: copy all headers to GUBG::shared')
    puts('define: build and copy libraries and executables to GUBG::shared')
end

task :declare do
    publish('src', pattern: '**/*.hpp', dst: 'include')
end

task :define => :declare do
    lib = Build::Library.new('gubg.ml')
    lib.add_include_path(shared_dir('include'))
    lib.add_sources(FileList.new('src/gubg/**/*.cpp'))
    lib.add_sources(FileList.new(shared('include', '**/*.hpp')))
    lib.build
    publish(lib.lib_filename, dst: 'lib')
end

task :test => :define do
    Rake::Task['ut:test'].invoke
end

namespace :ut do
    ut = nil
    task :setup do
        ut = Build::Executable.new('unit_tests')
        ut.add_include_path(shared_dir('include'))
        ut.add_define('DEBUG')
        ut.add_sources(FileList.new('src/test/**/*.cpp'))
        ut.add_sources(FileList.new(shared('include', '**/*.hpp')))
        ut.add_sources(shared_file('source', 'catch_runner.cpp'))
        ut.add_library_path(shared_dir('lib'))
        ut.add_library('gubg.ml')
    end
    task :test => :setup do
        ut.build
        options = %w[-d yes -a]
        options << '[ut]'
        ut.run(options)
    end
end
