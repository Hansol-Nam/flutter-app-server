import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:camera/camera.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image/image.dart' as img;
import 'package:permission_handler/permission_handler.dart';
import 'package:path_provider/path_provider.dart';
import 'package:gallery_saver/gallery_saver.dart';
import 'dart:io';

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
      enableContours: true,
      enableClassification: true,
      minFaceSize: 0.1, // 최소 얼굴 크기 비율
      performanceMode: FaceDetectorMode.accurate, // 정확도 우선 모드
    ),
  );
  bool _isDetecting = false;
  List<Face> _faces = [];
  String _emotion = "알 수 없음";

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _requestPermission();
    _controller = CameraController(
      widget.camera,
      ResolutionPreset.high,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420, // YUV420 강제 설정
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
    if (_isDetecting) return;
    _isDetecting = true;

    try {
      final Size imageSize =
          Size(image.width.toDouble(), image.height.toDouble());

      // 카메라의 센서 방향을 기반으로 InputImageRotation 설정
      final imageRotation = _convertSensorOrientationToInputImageRotation(
          widget.camera.sensorOrientation);

      final inputImageFormat =
          InputImageFormatValue.fromRawValue(image.format.raw) ??
              InputImageFormat.nv21;

      final planeData = image.planes.map(
        (Plane plane) {
          return InputImagePlaneMetadata(
            bytesPerRow: plane.bytesPerRow,
            height: plane.height,
            width: plane.width,
          );
        },
      ).toList();

      final inputImageData = InputImageData(
        size: imageSize,
        imageRotation: imageRotation,
        inputImageFormat: inputImageFormat,
        planeData: planeData,
      );

      final inputImage = InputImage.fromBytes(
          bytes: _concatenatePlanes(image.planes),
          inputImageData: inputImageData);

      final faces = await _faceDetector.processImage(inputImage);

      if (faces.isEmpty) {
        print("No faces detected in the frame.");
        _isDetecting = false;
        return;
      }

      if (mounted) {
        setState(() {
          _faces = faces;
        });
      }

      for (var face in faces) {
        // 얼굴의 boundingBox 추출
        final rect = face.boundingBox;

        // 카메라 프레임에서 얼굴 영역 추출
        Uint8List? faceBytes = await _extractFaceBytes(image, rect);

        if (faceBytes != null) {
          // 네이티브 코드로 얼굴 이미지 전송 및 감정 예측
          String emotion = await _getEmotionFromServer(faceBytes);
          setState(() {
            _emotion = emotion;
          });
        }
      }

      // 디버깅: 감지된 얼굴 수와 바운딩 박스 정보 출력
      print("Detected ${faces.length} face(s).");
      for (var face in faces) {
        print("Face boundingBox: ${face.boundingBox}");
      }
    } catch (e) {
      print("Error processing image: $e");
    } finally {
      _isDetecting = false;
    }
  }

  // 이미지의 모든 평면을 하나의 바이트 배열로 결합
  Uint8List _concatenatePlanes(List<Plane> planes) {
    final WriteBuffer allBytes = WriteBuffer();
    for (Plane plane in planes) {
      allBytes.putUint8List(plane.bytes);
    }
    return allBytes.done().buffer.asUint8List();
  }

  Future<Uint8List?> _extractFaceBytes(
      CameraImage image, Rect boundingBox) async {
    try {
      print("Starting to process bounding box: $boundingBox");
      final img.Image convertedImage = _convertYUV420ToImage(image);

      // 변환된 이미지 크기
      final int imgWidth = convertedImage.width;
      final int imgHeight = convertedImage.height;

      print("Converted image size: $imgWidth x $imgHeight");
      // 좌표 변환
      final double scaleX = imgWidth / image.width;
      final double scaleY = imgHeight / image.height;

      final int left =
          (boundingBox.left * scaleX).clamp(0, imgWidth - 1).toInt();
      final int top =
          (boundingBox.top * scaleY).clamp(0, imgHeight - 1).toInt();
      final int right =
          (boundingBox.right * scaleX).clamp(0, imgWidth - 1).toInt();
      final int bottom =
          (boundingBox.bottom * scaleY).clamp(0, imgHeight - 1).toInt();

      if (left < 0 || top < 0 || right <= left || bottom <= top) {
        print("Invalid transformed bounding box. Skipping...");
        return null;
      }

      // Bounding Box 크기 계산
      final int width = (right - left).clamp(1, imgWidth).toInt();
      final int height = (bottom - top).clamp(1, imgHeight).toInt();

      // 디버깅: 좌표 및 크기 출력
      print(
          "Transformed Rect: left=$left, top=$top, width=$width, height=$height");

      if (width <= 0 || height <= 0) {
        print("Invalid bounding box dimensions: width=$width, height=$height");
        return null;
      }

      // 얼굴 영역 자르기
      final img.Image faceImage =
          img.copyCrop(convertedImage, left, top, width, height);

      // 자른 이미지를 JPEG로 인코딩
      final Uint8List faceBytes = Uint8List.fromList(img.encodePng(faceImage));

      // 갤러리에 저장
      await saveImageToGallery(
          faceBytes, "face_debug_${DateTime.now().millisecondsSinceEpoch}");
      print("Face bytes extracted successfully.");

      return faceBytes;
    } catch (e) {
      print("Error extracting face bytes: $e");
      return null;
    }
  }

  // YUV420을 RGB로 변환하는 메서드
  img.Image _convertYUV420ToImage(CameraImage image) {
    try {
      final int width = image.width;
      final int height = image.height;

      print("Converting YUV420 image with size: ${width}x$height");
      print("Image planes length: ${image.planes.length}");

      if (image.planes.length == 2) {
        print("Detected NV21/NV12 format.");
        final img.Image imgImage = img.Image(width, height);

        final Uint8List yPlane = image.planes[0].bytes;
        final Uint8List uvPlane = image.planes[1].bytes;

        for (int y = 0; y < height; y++) {
          for (int x = 0; x < width; x++) {
            final int yIndex = y * image.planes[0].bytesPerRow + x;
            final int uvIndex =
                (y ~/ 2) * image.planes[1].bytesPerRow + (x ~/ 2) * 2;

            final int yValue = yPlane[yIndex];
            final int uValue = uvPlane[uvIndex + 0];
            final int vValue = uvPlane[uvIndex + 1];

            final int r =
                (yValue + 1.402 * (vValue - 128)).clamp(0, 255).toInt();
            final int g =
                (yValue - 0.344136 * (uValue - 128) - 0.714136 * (vValue - 128))
                    .clamp(0, 255)
                    .toInt();
            final int b =
                (yValue + 1.772 * (uValue - 128)).clamp(0, 255).toInt();

            imgImage.setPixelRgba(x, y, r, g, b);
          }
        }

        print("NV21/NV12 to RGB conversion complete.");
        return imgImage;
      }

      throw Exception("Unsupported planes length: ${image.planes.length}");
    } catch (e, stackTrace) {
      print("Error in YUV420 to RGB conversion: $e");
      print("Stack trace: $stackTrace");
      rethrow;
    }
  }

  // 서버와 통신하여 감정 예측
  Future<String> _getEmotionFromServer(Uint8List faceBytes) async {
    try {
      //final String emotion = await platform.invokeMethod('getEmotion', {
      //  'faceBytes': faceBytes,
      //});
      final String emotion = "test";
      print('Emotion detected: $emotion');
      return emotion;
    } on PlatformException catch (e) {
      print("Failed to get emotion: '${e.message}'.");
      return "오류 발생";
    }
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

  @override
  Widget build(BuildContext context) {
    if (!_controller.value.isInitialized) {
      return Scaffold(
        appBar: AppBar(title: const Text('Face Emotion Recognition')),
        body: const Center(child: CircularProgressIndicator()),
      );
    }
    return Scaffold(
      appBar: AppBar(title: const Text('Face Emotion Recognition')),
      body: Stack(
        fit: StackFit.expand,
        children: [
          AspectRatio(
            aspectRatio: _controller.value.aspectRatio,
            child: CameraPreview(_controller),
          ),
          // 바운딩 박스 그리기
          CustomPaint(
            painter: FacePainter(
              faces: _faces,
              imageSize: _controller.value.previewSize!,
              imageRotation: _convertSensorOrientationToInputImageRotation(
                  widget.camera.sensorOrientation),
              isFrontCamera:
                  widget.camera.lensDirection == CameraLensDirection.front,
            ),
          ),
          // 감정 결과 표시
          Positioned(
            bottom: 50,
            left: 0,
            right: 0,
            child: Center(
              child: Container(
                padding: const EdgeInsets.all(10),
                color: Colors.black54,
                child: Text(
                  'Emotion: $_emotion',
                  style: const TextStyle(color: Colors.white, fontSize: 20),
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
  final InputImageRotation imageRotation;
  final bool isFrontCamera;

  FacePainter({
    required this.faces,
    required this.imageSize,
    required this.imageRotation,
    required this.isFrontCamera,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.redAccent
      ..strokeWidth = 2.0
      ..style = PaintingStyle.stroke;

    // 이미지의 회전에 따라 스케일링을 조정
    double scaleX, scaleY;

    switch (imageRotation) {
      case InputImageRotation.rotation90deg:
      case InputImageRotation.rotation270deg:
        // 이미지가 가로로 회전된 경우
        scaleX = size.width / imageSize.height;
        scaleY = size.height / imageSize.width;
        break;
      default:
        // 이미지가 세로로 회전된 경우
        scaleX = size.width / imageSize.width;
        scaleY = size.height / imageSize.height;
    }

    for (var face in faces) {
      final rect = face.boundingBox;

      // 이미지 회전에 따라 좌표를 조정
      Rect transformedRect;
      switch (imageRotation) {
        case InputImageRotation.rotation90deg:
          transformedRect = Rect.fromLTRB(
            rect.top * scaleX,
            rect.left * scaleY,
            rect.bottom * scaleX,
            rect.right * scaleY,
          );
          break;
        case InputImageRotation.rotation270deg:
          transformedRect = Rect.fromLTRB(
            size.width - rect.bottom * scaleX,
            rect.left * scaleY,
            size.width - rect.top * scaleX,
            rect.right * scaleY,
          );
          break;
        case InputImageRotation.rotation180deg:
          transformedRect = Rect.fromLTRB(
            size.width - rect.right * scaleX,
            size.height - rect.bottom * scaleY,
            size.width - rect.left * scaleX,
            size.height - rect.top * scaleY,
          );
          break;
        default:
          transformedRect = Rect.fromLTRB(
            rect.left * scaleX,
            rect.top * scaleY,
            rect.right * scaleX,
            rect.bottom * scaleY,
          );
      }

      // 프론트 카메라일 경우 좌우 반전
      if (isFrontCamera) {
        transformedRect = Rect.fromLTRB(
          size.width - transformedRect.right,
          transformedRect.top,
          size.width - transformedRect.left,
          transformedRect.bottom,
        );
      }

      // 디버깅: 변환된 바운딩 박스 좌표 출력
      print("Transformed Rect: $transformedRect");

      // 바운딩 박스 그리기
      canvas.drawRect(transformedRect, paint);
    }
  }

  @override
  bool shouldRepaint(FacePainter oldDelegate) {
    return oldDelegate.faces != faces ||
        oldDelegate.imageSize != imageSize ||
        oldDelegate.imageRotation != imageRotation ||
        oldDelegate.isFrontCamera != isFrontCamera;
  }
}

Future<void> saveImageToGallery(Uint8List imageBytes, String fileName) async {
  try {
    // 임시 디렉토리 경로 가져오기
    final directory = await getTemporaryDirectory();
    final imagePath = '${directory.path}/$fileName.png';

    // 파일로 저장
    final file = File(imagePath);
    await file.writeAsBytes(imageBytes);

    print("Image saved to temporary path: $imagePath");

    // 갤러리에 저장
    await GallerySaver.saveImage(imagePath);
    print("Image saved to gallery.");
  } catch (e) {
    print("Error saving image to gallery: $e");
  }
}
