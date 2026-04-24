package com.example.autocropdemo.engine

import android.graphics.Bitmap
import android.graphics.Rect
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Rect as CvRect
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import kotlin.math.max

class AutoCropEngine {

    fun autoCrop(inputBitmap: Bitmap): CropResult {
        val rgba = Mat()
        Utils.bitmapToMat(inputBitmap, rgba)

        if (rgba.type() != CvType.CV_8UC4) {
            Imgproc.cvtColor(rgba, rgba, Imgproc.COLOR_BGR2RGBA)
        }

        val h = rgba.rows()
        val w = rgba.cols()
        val border = max(4, (minOf(w, h) * 0.04).toInt())

        val bgMeanLab = estimateBorderBackgroundLab(rgba, border)

        val rgb = Mat()
        Imgproc.cvtColor(rgba, rgb, Imgproc.COLOR_RGBA2RGB)

        val lab = Mat()
        Imgproc.cvtColor(rgb, lab, Imgproc.COLOR_RGB2Lab)

        val bg = Mat(lab.size(), lab.type(), bgMeanLab)
        val diff = Mat()
        Core.absdiff(lab, bg, diff)

        val channels = ArrayList<Mat>(3)
        Core.split(diff, channels)
        val diffGray = Mat()
        Core.addWeighted(channels[0], 1.0, channels[1], 1.0, 0.0, diffGray)
        Core.addWeighted(diffGray, 1.0, channels[2], 1.0, 0.0, diffGray)

        val mask = Mat()
        Imgproc.threshold(diffGray, mask, 0.0, 255.0, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU)

        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, org.opencv.core.Size(5.0, 5.0))
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_OPEN, kernel)
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, kernel)

        val contourRect = largestContourRect(mask) ?: CvRect(0, 0, w, h)
        val safeRect = clampRect(contourRect, w, h)

        val croppedRgba = Mat(rgba, safeRect)
        val maskCrop = Mat(mask, safeRect)

        val out = Mat.zeros(croppedRgba.size(), CvType.CV_8UC4)
        croppedRgba.copyTo(out, maskCrop)

        val resultBitmap = Bitmap.createBitmap(out.cols(), out.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(out, resultBitmap)

        releaseAll(
            rgba, rgb, lab, bg, diff, diffGray, mask, kernel, croppedRgba, maskCrop, out,
            *channels.toTypedArray()
        )

        return CropResult(
            bitmap = resultBitmap,
            rect = Rect(safeRect.x, safeRect.y, safeRect.x + safeRect.width, safeRect.y + safeRect.height),
            debugInfo = "rect=${safeRect.width}x${safeRect.height} @ (${safeRect.x},${safeRect.y})"
        )
    }

    private fun estimateBorderBackgroundLab(rgba: Mat, b: Int): Scalar {
        val h = rgba.rows()
        val w = rgba.cols()

        val top = Mat(rgba, CvRect(0, 0, w, b))
        val bottom = Mat(rgba, CvRect(0, h - b, w, b))
        val left = Mat(rgba, CvRect(0, 0, b, h))
        val right = Mat(rgba, CvRect(w - b, 0, b, h))

        val strips = listOf(top, bottom, left, right)

        val labMeans = strips.map {
            val lab = Mat()
            val rgb = Mat()
            Imgproc.cvtColor(it, rgb, Imgproc.COLOR_RGBA2RGB)
            Imgproc.cvtColor(rgb, lab, Imgproc.COLOR_RGB2Lab)
            val m = Core.mean(lab)
            rgb.release()
            lab.release()
            m
        }

        releaseAll(top, bottom, left, right)

        val l = labMeans.map { it.`val`[0] }.average()
        val a = labMeans.map { it.`val`[1] }.average()
        val bb = labMeans.map { it.`val`[2] }.average()

        return Scalar(l, a, bb, 0.0)
    }

    private fun largestContourRect(mask: Mat): CvRect? {
        val contours = mutableListOf<org.opencv.core.MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(mask.clone(), contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        var bestArea = 0.0
        var best: CvRect? = null
        for (c in contours) {
            val area = Imgproc.contourArea(c)
            if (area > bestArea) {
                bestArea = area
                best = Imgproc.boundingRect(c)
            }
            c.release()
        }
        hierarchy.release()
        return best
    }

    private fun clampRect(r: CvRect, w: Int, h: Int): CvRect {
        val x = r.x.coerceIn(0, w - 1)
        val y = r.y.coerceIn(0, h - 1)
        val width = r.width.coerceIn(1, w - x)
        val height = r.height.coerceIn(1, h - y)
        return CvRect(Point(x.toDouble(), y.toDouble()), Point((x + width).toDouble(), (y + height).toDouble()))
    }

    private fun releaseAll(vararg mats: Mat) {
        mats.forEach { it.release() }
    }
}
