use_frameworks!
inhibit_all_warnings!

target 'OSX' do
  platform :osx, '10.10'
  link_with 'BrainCore.OSX', 'BrainCoreTests.OSX'
  pod 'Upsurge', :git => 'https://github.com/aleph7/Upsurge.git', :branch => 'master'
end

target 'iOS' do
  platform :ios, '8.4'
  link_with 'BrainCore.iOS', 'BrainCoreTests.iOS'
  pod 'Upsurge', :git => 'https://github.com/aleph7/Upsurge.git', :branch => 'master'
end
