import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:camera/camera.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image/image.dart' as img;
import 'package:permission_handler/permission_handler.dart';
import 'dart:math' as math;
import 'package:http/http.dart' as http;
import 'dart:convert'; // jsonDecode를 위해 추가
import 'dart:isolate';
import 'package:path_provider/path_provider.dart'; // 추가
import 'package:http_parser/http_parser.dart'; // MediaType을 위해 필요

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // 사용 가능한 카메라 목록 가져오기
  final cameras = await availableCameras();
  final firstCamera = cameras.first;

  runApp(MyApp(camera: firstCamera));
}

class MyApp extends StatelessWidget {
  final CameraDescription camera;

  const MyApp({super.key, required this.camera});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Face Emotion Recognition',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: FaceDetectionPage(camera: camera),
    );
  }
}

class FaceDetectionPage extends StatefulWidget {
  final CameraDescription camera;

  const FaceDetectionPage({super.key, required this.camera});

  @override
  _FaceDetectionPageState createState() => _FaceDetectionPageState();
}

class _FaceDetectionPageState extends State<FaceDetectionPage>
    with WidgetsBindingObserver {
  late CameraController _controller;
  late Future<void> _initializeControllerFuture;
  final FaceDetector _faceDetector = FaceDetector(
    options: FaceDetectorOptions(
      enableContours: false,
      enableClassification: false,
      minFaceSize: 0.1,
      performanceMode: FaceDetectorMode.fast,
    ),
  );
  //bool _isDetecting = false;
  List<Face> _faces = [];
  String _emotion = "알 수 없음";

  DateTime? _lastProcessedTime;
  DateTime? _lastEmotionTime;

  static const Duration frameInterval =
      Duration(milliseconds: 50); // 프레임 처리 간격 조정
  static const Duration emotionInterval = Duration(seconds: 2); // 감정 예측 간격 조정

  String imgName = "test_image";

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _requestPermission();
    _controller = CameraController(
      widget.camera,
      ResolutionPreset.low, // 해상도 낮춤으로써 처리량 감소
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.bgra8888, // iOS에서는 bgra8888 사용
    );
    _initializeControllerFuture = _controller.initialize().then((_) {
      if (!mounted) return;
      setState(() {});
      _controller.startImageStream(_processCameraImage);
    });
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _controller.dispose();
    _faceDetector.close();
    super.dispose();
  }

  // 권한 요청
  Future<void> _requestPermission() async {
    var status = await Permission.camera.status;
    if (!status.isGranted) {
      await Permission.camera.request();
    }
  }

  // 카메라 프레임 처리
  void _processCameraImage(CameraImage image) async {
    final now = DateTime.now();

    if (_lastProcessedTime != null &&
        now.difference(_lastProcessedTime!) < frameInterval) {
      return;
    }
    _lastProcessedTime = now;

    //if (_isDetecting) return;
    //_isDetecting = true;

    try {
      // `CameraImage`를 `InputImage`로 변환
      final inputImage = _getInputImageFromCameraImage(image);

      // 메인 스레드에서 얼굴 감지 수행
      final faces = await _faceDetector.processImage(inputImage);

      if (mounted) {
        setState(() {
          _faces = faces;
        });
      }

      if (faces.isNotEmpty) {
        final faceRect = faces.first.boundingBox;

        // 얼굴 영역 추출
        final Uint8List faceBytes = _extractFaceBytes(image, faceRect);

        final now = DateTime.now();
        if (_lastEmotionTime == null ||
            now.difference(_lastEmotionTime!) >= emotionInterval) {
          _lastEmotionTime = now;

          // 감정 예측 수행
          String emotion = await _getEmotionFromServer(faceBytes);
          if (mounted) {
            setState(() {
              _emotion = emotion;
            });
          }
        }
      }
    } catch (e) {
      print("Error processing image: $e");
    } finally {
      //  _isDetecting = false;
    }
  }

  InputImage _getInputImageFromCameraImage(CameraImage image) {
    final allBytes = WriteBuffer();
    for (var plane in image.planes) {
      allBytes.putUint8List(plane.bytes);
    }
    final bytes = allBytes.done().buffer.asUint8List();

    final inputImage = InputImage.fromBytes(
      bytes: bytes,
      inputImageData: InputImageData(
        size: Size(image.width.toDouble(), image.height.toDouble()),
        imageRotation: _convertSensorOrientationToInputImageRotation(
            widget.camera.sensorOrientation),
        inputImageFormat: InputImageFormat.bgra8888,
        planeData: image.planes.map((Plane plane) {
          return InputImagePlaneMetadata(
            bytesPerRow: plane.bytesPerRow,
            height: plane.height,
            width: plane.width,
          );
        }).toList(),
      ),
    );

    return inputImage;
  }

  Uint8List _extractFaceBytes(CameraImage image, Rect boundingBox) {
    // `CameraImage`에서 얼굴 영역을 추출하고, JPEG로 인코딩하여 `Uint8List`로 반환합니다.
    // `image`는 `bgra8888` 포맷이므로, 이미지 변환 없이 처리 가능합니다.

    // 이미지의 전체 크기
    final int imageWidth = image.width;
    final int imageHeight = image.height;

    // 바운딩 박스 좌표 계산
    final int left = boundingBox.left.toInt().clamp(0, imageWidth - 1);
    final int top = boundingBox.top.toInt().clamp(0, imageHeight - 1);
    final int width = boundingBox.width.toInt().clamp(1, imageWidth - left);
    final int height = boundingBox.height.toInt().clamp(1, imageHeight - top);

    // 얼굴 영역의 바이트 배열 추출
    final Uint8List faceBytes = _cropCameraImage(
      image,
      left,
      top,
      width,
      height,
    );

    // 얼굴 이미지를 JPEG로 인코딩
    final img.Image faceImage = img.Image.fromBytes(
      width,
      height,
      faceBytes,
      format: img.Format.bgra,
    );

    return Uint8List.fromList(img.encodeJpg(faceImage));
  }

  Uint8List _cropCameraImage(
      CameraImage image, int x, int y, int width, int height) {
    // `bgra8888` 포맷의 `CameraImage`에서 특정 영역을 추출합니다.

    final int bytesPerPixel = image.planes[0].bytesPerPixel ?? 4;
    final int bytesPerRow = image.planes[0].bytesPerRow;

    final Uint8List croppedBytes = Uint8List(width * height * bytesPerPixel);

    for (int row = 0; row < height; row++) {
      final int srcOffset = (row + y) * bytesPerRow + x * bytesPerPixel;
      final int destOffset = row * width * bytesPerPixel;

      croppedBytes.setRange(
        destOffset,
        destOffset + width * bytesPerPixel,
        image.planes[0].bytes,
        srcOffset,
      );
    }

    return croppedBytes;
  }

  // 센서 방향을 ML Kit의 InputImageRotation으로 변환
  InputImageRotation _convertSensorOrientationToInputImageRotation(
      int sensorOrientation) {
    switch (sensorOrientation) {
      case 0:
        return InputImageRotation.rotation0deg;
      case 90:
        return InputImageRotation.rotation90deg;
      case 180:
        return InputImageRotation.rotation180deg;
      case 270:
        return InputImageRotation.rotation270deg;
      default:
        return InputImageRotation.rotation0deg;
    }
  }

  // 서버와 통신하여 감정 예측
  Future<String> _getEmotionFromServer(Uint8List faceBytes) async {
    try {
      print("Starting server request...");
      final requestStartTime = DateTime.now();
      //final String emotion = await platform.invokeMethod('getEmotion', {
      //  'faceBytes': faceBytes,
      //});
      String _emotion = "알수없음";
      String _result = "";

      final uri =
          Uri.parse("http://34.64.124.155:8000/analyze"); // 서버 주소와 포트 입력
      final request = http.MultipartRequest("POST", uri);

      // 바이트 배열을 바로 전송
      request.files.add(http.MultipartFile.fromBytes(
        'file', // FastAPI에서 지정한 필드 이름
        faceBytes,
        filename: '$imgName.png', // 파일 이름 지정
        contentType: MediaType('image', 'png'), // 이미지 타입 지정
      ));
      // 요청 보내기
      final response = await request.send();

      // 응답 처리
      if (response.statusCode == 200) {
        final responseData = await response.stream.bytesToString();
        final data = jsonDecode(responseData);

        // JSON 응답에서 emotion과 logits 추출
        if (data['status'] == 'success' && data['result'] is Map) {
          final result = data['result'] as Map<String, dynamic>;
          final emotion = result['emotion'] as String;
          final logits = result['logits'] as List<dynamic>;

          _result = "Emotion: $emotion\nLogits: $logits";
          _emotion = emotion;
        } else {
          _result = "Error : GetEmotionFromServer Fail";
        }
      } else {
        if (mounted) {
          setState(() {
            _result = "Error: ${response.statusCode}";
          });
        }
      }
      //print('result: $_result');

      final requestEndTime = DateTime.now();
      final requestDuration = requestEndTime.difference(requestStartTime);
      print("Server request took ${requestDuration.inMilliseconds} ms");

      return _emotion;
    } catch (e) {
      print("Error uploading image: $e");
      return "알수없음";
    }
  }

  @override
  Widget build(BuildContext context) {
    if (!_controller.value.isInitialized) {
      return Scaffold(
        appBar: AppBar(title: const Text('Face Emotion Recognition')),
        body: const Center(child: CircularProgressIndicator()),
      );
    }

    // 화면의 크기 가져오기
    final Size screenSize = MediaQuery.of(context).size;

    return Scaffold(
      appBar: AppBar(title: const Text('Face Emotion Recognition')),
      body: Stack(
        fit: StackFit.expand,
        children: [
          CameraPreview(_controller),
          // 바운딩 박스 그리기
          CustomPaint(
            painter: FacePainter(
              faces: _faces,
              imageSize: Size(
                _controller.value.previewSize!.height,
                _controller.value.previewSize!.width,
              ),
              cameraLensDirection: widget.camera.lensDirection,
            ),
          ),
          // 감정 결과 표시
          Positioned(
            bottom: 50,
            left: 0,
            right: 0,
            child: Center(
              child: Container(
                padding: EdgeInsets.all(10),
                color: Colors.black54,
                child: Text(
                  'Emotion: $_emotion',
                  style: TextStyle(color: Colors.white, fontSize: 20),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class FacePainter extends CustomPainter {
  final List<Face> faces;
  final Size imageSize;
  final CameraLensDirection cameraLensDirection;

  FacePainter({
    required this.faces,
    required this.imageSize,
    required this.cameraLensDirection,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final Paint paint = Paint()
      ..color = Colors.redAccent
      ..strokeWidth = 2.0
      ..style = PaintingStyle.stroke;

    // 이미지와 화면의 종횡비 계산
    double imageAspectRatio = imageSize.height / imageSize.width;
    double canvasAspectRatio = size.height / size.width;

    double scaleX, scaleY;
    double dx = 0, dy = 0;

    if (canvasAspectRatio > imageAspectRatio) {
      // 화면이 더 길쭉한 경우
      scaleX = size.width / imageSize.width;
      scaleY = scaleX;
      double scaledImageHeight = imageSize.height * scaleY;
      dy = (size.height - scaledImageHeight) / 2;
    } else {
      // 이미지가 더 길쭉한 경우
      scaleY = size.height / imageSize.height;
      scaleX = scaleY;
      double scaledImageWidth = imageSize.width * scaleX;
      dx = (size.width - scaledImageWidth) / 2;
    }

    for (Face face in faces) {
      Rect rect = face.boundingBox;

      // 좌표 변환
      rect = Rect.fromLTRB(
        rect.left * scaleX + dx,
        rect.top * scaleY + dy,
        rect.right * scaleX + dx,
        rect.bottom * scaleY + dy,
      );

      // 프론트 카메라일 경우 좌우 반전
      if (cameraLensDirection == CameraLensDirection.front) {
        rect = Rect.fromLTRB(
          size.width - rect.right,
          rect.top,
          size.width - rect.left,
          rect.bottom,
        );
      }

      // 바운딩 박스 그리기
      canvas.drawRect(rect, paint);
    }
  }

  @override
  bool shouldRepaint(FacePainter oldDelegate) {
    return oldDelegate.faces != faces ||
        oldDelegate.imageSize != imageSize ||
        oldDelegate.cameraLensDirection != cameraLensDirection;
  }
}
