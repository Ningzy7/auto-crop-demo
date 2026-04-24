# AutoCropDemo

Android Kotlin + OpenCV 示例项目：
- 从相册选择图片
- 按形状/色差自动识别主体区域
- 自动裁切并保留透明背景
- 导出 PNG 到系统相册

## 技术点
- Kotlin + View XML
- OpenCV 4.9（Maven）
- 轮廓检测 + 边界色差前景分离

## 目录
- `app/src/main/java/com/example/autocropdemo/MainActivity.kt`
- `app/src/main/java/com/example/autocropdemo/engine/AutoCropEngine.kt`

## 说明
这是 Demo 版本，目标是快速验证自动裁切效果，后续可加：
- GrabCut 精修
- AI 分割（TFLite）
- 手动涂抹修边
