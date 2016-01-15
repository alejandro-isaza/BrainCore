Pod::Spec.new do |s|
  s.name         = "BrainCore"
  s.version      = "0.0.3"
  s.summary      = "The iOS and OS X neural network framework"
  s.homepage     = "https://github.com/aleph7/BrainCore"
  s.license      = "MIT"
  s.author       = { "Alejandro Isaza" => "al@isaza.ca" }
  
  s.ios.deployment_target = "8.4"
  s.osx.deployment_target = "10.10"
  # s.tvos.deployment_target = "9.0"

  s.source       = { git: "https://github.com/aleph7/BrainCore.git", tag: s.version, submodules: true }
  s.source_files  = "Source", "Source/Layers"

  s.dependency "Upsurge", '~> 0.4.6'
end
