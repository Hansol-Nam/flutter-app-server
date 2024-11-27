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
import 'dart:convert';
import 'package:path_provider/path_provider.dart';
import 'package:http_parser/http_parser.dart';
import 'package:flutter/foundation.dart'; // compute 함수를 사용하기 위해 추가

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

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
  List<Face> _faces = [];
  String _emotion = "알 수 없음";

  DateTime? _lastProcessedTime;
  DateTime? _lastEmotionTime;

  static const Duration frameInterval =
      Duration(milliseconds: 50); // 프레임 처리 간격 조정
  static const Duration emotionInterval =
      Duration(seconds: 3); // 감정 예측 간격 조정 (조절 가능)

  String imgName = "test_image";

  bool _isProcessing = false; // 새로운 상태 변수

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

    if (_isProcessing) return; // 이미 처리 중이면 새 요청 방지

    try {
      final inputImage = _getInputImageFromCameraImage(image);
      final faces = await _faceDetector.processImage(inputImage);

      if (mounted) {
        setState(() {
          _faces = faces;
        });
      }

      if (faces.isNotEmpty) {
        final faceRect = faces.first.boundingBox;

        final Uint8List faceBytes = await _extractFaceBytes(image, faceRect);

        final now = DateTime.now();
        if (_lastEmotionTime == null ||
            now.difference(_lastEmotionTime!) >= emotionInterval) {
          _lastEmotionTime = now;

          String emotion = await _getEmotionFromServer(faceBytes);
          if (mounted) {
            setState(() {
              _emotion = emotion;
            });
          }
        }
      }
    } catch (e) {
      print("이미지 처리 오류: $e");
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

  // 이미지 변환을 별도의 Isolate에서 처리
  static Uint8List _convertImage(List<dynamic> params) {
    Uint8List faceBytes = params[0];
    int width = params[1];
    int height = params[2];

    final img.Image faceImage = img.Image.fromBytes(
      width,
      height,
      faceBytes,
      format: img.Format.bgra,
    );

    // 이미지 크기 축소 (예: 50% 축소)
    final img.Image resizedImage = img.copyResize(
      faceImage,
      width: (faceImage.width * 0.5).toInt(),
      height: (faceImage.height * 0.5).toInt(),
    );

    return Uint8List.fromList(
        img.encodeJpg(resizedImage, quality: 80)); // 품질 조정
  }

  Future<Uint8List> _extractFaceBytes(
      CameraImage image, Rect boundingBox) async {
    final int imageWidth = image.width;
    final int imageHeight = image.height;

    final int left = boundingBox.left.toInt().clamp(0, imageWidth - 1);
    final int top = boundingBox.top.toInt().clamp(0, imageHeight - 1);
    final int width = boundingBox.width.toInt().clamp(1, imageWidth - left);
    final int height = boundingBox.height.toInt().clamp(1, imageHeight - top);

    final Uint8List faceBytes = _cropCameraImage(
      image,
      left,
      top,
      width,
      height,
    );

    // compute 함수를 사용하여 별도의 Isolate에서 이미지 변환 수행
    final Uint8List resizedFaceBytes =
        await compute(_convertImage, [faceBytes, width, height]);

    return resizedFaceBytes;
  }

  Uint8List _cropCameraImage(
      CameraImage image, int x, int y, int width, int height) {
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
    setState(() {
      _isProcessing = true; // 처리 시작
      _emotion = "처리 중..."; // 처리 중임을 표시
      //print("처리 시작: _isProcessing = $_isProcessing");
    });

    try {
      //print("서버 요청 시작...");
      final requestStartTime = DateTime.now();

      String emotion = "알 수 없음";

      final uri = Uri.parse("http://34.64.124.155:8000/analyze"); // 서버 주소와 포트
      final request = http.MultipartRequest("POST", uri);

      request.files.add(http.MultipartFile.fromBytes(
        'file', // 서버에서 기대하는 필드 이름
        faceBytes,
        filename: '$imgName.png', // 파일 이름
        contentType: MediaType('image', 'png'), // 이미지 타입 지정
      ));

      final response = await request.send();

      if (response.statusCode == 200) {
        final responseData = await response.stream.bytesToString();
        final data = jsonDecode(responseData);

        if (data['status'] == 'success' && data['result'] is Map) {
          final resultData = data['result'] as Map<String, dynamic>;
          emotion = resultData['emotion'] as String;
        } else {
          print("서버 응답 오류: 감정 분석 실패");
        }
      } else {
        print("서버 응답 오류: 상태 코드 ${response.statusCode}");
      }

      final requestEndTime = DateTime.now();
      final requestDuration = requestEndTime.difference(requestStartTime);
      print("서버 요청 시간: ${requestDuration.inMilliseconds} ms");

      return emotion;
    } catch (e) {
      print("이미지 업로드 오류: $e");
      return "알 수 없음";
    } finally {
      setState(() {
        _isProcessing = false; // 처리 종료
        //print("처리 종료: _isProcessing = $_isProcessing");
      });
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

    final Size screenSize = MediaQuery.of(context).size;

    return Scaffold(
      appBar: AppBar(title: const Text('Face Emotion Recognition')),
      body: Stack(
        fit: StackFit.expand,
        children: [
          CameraPreview(_controller),
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
          // 처리 중 오버레이
          if (_isProcessing)
            Container(
              color: Colors.black45,
              child: Center(
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    CircularProgressIndicator(
                      valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                      strokeWidth: 2.0,
                    ),
                    SizedBox(width: 10),
                    Text(
                      '처리 중...',
                      style: TextStyle(color: Colors.white, fontSize: 20),
                    ),
                  ],
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
