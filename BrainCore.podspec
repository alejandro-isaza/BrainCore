Pod::Spec.new do |s|
  s.name         = "BrainCore"
  s.version      = "0.0.1"
  s.summary      = "The iOS and OS X neural network framework"
  s.homepage     = "https://github.com/aleph7/BrainCore"
  s.license      = "MIT"
  s.author             = { "Alejandro" => "email@address.com" }
  
  s.ios.deployment_target = "8.4"
  s.osx.deployment_target = "10.10"
  # s.tvos.deployment_target = "9.0"

  s.source       = { :git => "https://github.com/aleph7/BrainCore.git", :tag => "0.0.1" }
  s.source_files  = "BrainCore"

  s.dependency "Surge", :git => "https://github.com/aleph7/Surge.git"
end
