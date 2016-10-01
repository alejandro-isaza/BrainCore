Pod::Spec.new do |s|
  s.name         = "BrainCore"
  s.version      = "0.4.0"
  s.summary      = "The iOS and OS X neural network framework"
  s.homepage     = "https://github.com/aleph7/BrainCore"
  s.license      = "MIT"
  s.author       = { "Alejandro Isaza" => "al@isaza.ca" }
  
  s.ios.deployment_target = "9.0"
  s.osx.deployment_target = "10.11"

  s.source = { git: "https://github.com/aleph7/BrainCore.git", tag: s.version, submodules: true }
  s.source_files = "Source/**/*.swift", "Source/Metal/*.metal", "Source/Metal/*.h"
  s.private_header_files = "Source/Metal/*.h"

  s.dependency "Upsurge", '~> 0.8'
end
