$PROTOC  = "C:\Users\Season\.nuget\packages\grpc.tools\2.65.0\tools\windows_x64\protoc.exe"
$PLUGIN  = "C:\Users\Season\.nuget\packages\grpc.tools\2.65.0\tools\windows_x64\grpc_csharp_plugin.exe"
$INCLUDE = "C:\Users\Season\.nuget\packages\grpc.tools\2.65.0\build\native\include"

& $PROTOC `
  -I $INCLUDE `
  -I "D:\Season\PhysioLabXR\physiolabxr\scripting\attention_bci\vision_proto" `
  --csharp_out="C:\UnityProjects\ReNaSuite\Assets\Scenes\AIWingman\Protos\Generated" `
  --grpc_out="C:\UnityProjects\ReNaSuite\Assets\Scenes\AIWingman\Protos\Generated" `
  --plugin=protoc-gen-grpc="$PLUGIN" `
  "D:\Season\PhysioLabXR\physiolabxr\scripting\attention_bci\vision_proto\vision.proto"