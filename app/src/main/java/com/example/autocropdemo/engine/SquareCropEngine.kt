package com.example.autocropdemo.engine

import android.graphics.Bitmap
import android.graphics.Rect
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Rect as CvRect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

class SquareCropEngine {

    data class Candidate(
        val bitmap: Bitmap,
        val rect: Rect,
        val method: String,
        val score: Double,
        val debugInfo: String
    )

    fun detectThree(inputBitmap: Bitmap): List<Candidate> {
        val rgba = Mat()
        Utils.bitmapToMat(inputBitmap, rgba)
        val w = rgba.cols()
        val h = rgba.rows()

        val rects = listOf(
            detectByContour(rgba),
            detectByColorDiff(rgba),
            detectByProjection(rgba)
        )

        val candidates = rects.map { (method, rect, baseScore) ->
            val safe = clampSquare(rect, w, h)
            val out = Mat(rgba, safe)
            val bmp = Bitmap.createBitmap(out.cols(), out.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(out, bmp)
            out.release()
            Candidate(
                bitmap = bmp,
                rect = Rect(safe.x, safe.y, safe.x + safe.width, safe.y + safe.height),
                method = method,
                score = baseScore,
                debugInfo = "$method: ${safe.width}x${safe.height}@(${safe.x},${safe.y}), score=${"%.2f".format(baseScore)}"
            )
        }

        rgba.release()
        return candidates
    }

    private fun detectByContour(rgba: Mat): Triple<String, CvRect, Double> {
        val gray = Mat()
        Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.GaussianBlur(gray, gray, Size(5.0, 5.0), 0.0)

        val edges = Mat()
        Imgproc.Canny(gray, edges, 40.0, 120.0)
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(5.0, 5.0))
        Imgproc.morphologyEx(edges, edges, Imgproc.MORPH_CLOSE, kernel)

        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(edges.clone(), contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        val imageArea = rgba.cols() * rgba.rows().toDouble()
        var bestRect = fullCenterSquare(rgba.cols(), rgba.rows())
        var bestScore = 0.0

        for (c in contours) {
            val area = Imgproc.contourArea(c)
            if (area < imageArea * 0.03) {
                c.release()
                continue
            }
            val peri = Imgproc.arcLength(MatOfPoint2f(*c.toArray()), true)
            val approx = MatOfPoint2f()
            Imgproc.approxPolyDP(MatOfPoint2f(*c.toArray()), approx, 0.03 * peri, true)
            val r = Imgproc.boundingRect(c)
            val aspect = squareAspectScore(r)
            val areaRatio = (r.width * r.height).toDouble() / imageArea
            val polyBonus = if (approx.total().toInt() in 4..8) 0.25 else 0.0
            val score = aspect * 0.5 + areaReasonScore(areaRatio) * 0.25 + polyBonus
            if (score > bestScore) {
                bestScore = score
                bestRect = r
            }
            approx.release()
            c.release()
        }

        releaseAll(gray, edges, kernel, hierarchy)
        return Triple("轮廓方形", bestRect, bestScore.coerceAtLeast(0.25))
    }

    private fun detectByColorDiff(rgba: Mat): Triple<String, CvRect, Double> {
        val w = rgba.cols()
        val h = rgba.rows()
        val border = max(4, (min(w, h) * 0.04).toInt())
        val bgMean = estimateBorderMeanLab(rgba, border)

        val rgb = Mat()
        val lab = Mat()
        Imgproc.cvtColor(rgba, rgb, Imgproc.COLOR_RGBA2RGB)
        Imgproc.cvtColor(rgb, lab, Imgproc.COLOR_RGB2Lab)

        val bg = Mat(lab.size(), lab.type(), bgMean)
        val diff = Mat()
        Core.absdiff(lab, bg, diff)
        val channels = ArrayList<Mat>()
        Core.split(diff, channels)
        val diffGray = Mat()
        Core.addWeighted(channels[0], 1.0, channels[1], 1.0, 0.0, diffGray)
        Core.addWeighted(diffGray, 1.0, channels[2], 1.0, 0.0, diffGray)

        val mask = Mat()
        Imgproc.threshold(diffGray, mask, 0.0, 255.0, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU)
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(9.0, 9.0))
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, kernel)

        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(mask.clone(), contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        var union: CvRect? = null
        for (c in contours) {
            val r = Imgproc.boundingRect(c)
            val area = r.width * r.height
            if (area > w * h * 0.01) union = if (union == null) r else unionRect(union!!, r)
            c.release()
        }
        val baseRect = union ?: fullCenterSquare(w, h)
        val sq = rectToSquare(baseRect, w, h)
        val score = 0.35 + squareAspectScore(baseRect) * 0.25 + areaReasonScore((sq.width * sq.height).toDouble() / (w * h)) * 0.4

        releaseAll(rgb, lab, bg, diff, diffGray, mask, kernel, hierarchy, *channels.toTypedArray())
        return Triple("色差方形", sq, score)
    }

    private fun detectByProjection(rgba: Mat): Triple<String, CvRect, Double> {
        val w = rgba.cols()
        val h = rgba.rows()
        val gray = Mat()
        Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.GaussianBlur(gray, gray, Size(3.0, 3.0), 0.0)

        val gradX = Mat()
        val gradY = Mat()
        Imgproc.Sobel(gray, gradX, CvType.CV_16S, 1, 0)
        Imgproc.Sobel(gray, gradY, CvType.CV_16S, 0, 1)
        Core.convertScaleAbs(gradX, gradX)
        Core.convertScaleAbs(gradY, gradY)

        val colScores = DoubleArray(w)
        val rowScores = DoubleArray(h)
        for (x in 0 until w) colScores[x] = Core.sumElems(gradX.col(x)).`val`[0]
        for (y in 0 until h) rowScores[y] = Core.sumElems(gradY.row(y)).`val`[0]

        val marginX = max(2, (w * 0.03).toInt())
        val marginY = max(2, (h * 0.03).toInt())
        val left = maxIndex(colScores, marginX, w / 2)
        val right = maxIndex(colScores, w / 2, w - marginX)
        val top = maxIndex(rowScores, marginY, h / 2)
        val bottom = maxIndex(rowScores, h / 2, h - marginY)

        val raw = CvRect(left, top, max(1, right - left), max(1, bottom - top))
        val sq = rectToSquare(raw, w, h)
        val aspect = squareAspectScore(raw)
        val score = 0.25 + aspect * 0.35 + areaReasonScore((sq.width * sq.height).toDouble() / (w * h)) * 0.4

        releaseAll(gray, gradX, gradY)
        return Triple("投影方形", sq, score)
    }

    private fun estimateBorderMeanLab(rgba: Mat, b: Int): Scalar {
        val h = rgba.rows()
        val w = rgba.cols()
        val rects = listOf(CvRect(0, 0, w, b), CvRect(0, h - b, w, b), CvRect(0, 0, b, h), CvRect(w - b, 0, b, h))
        val means = rects.map { r ->
            val roi = Mat(rgba, r)
            val rgb = Mat()
            val lab = Mat()
            Imgproc.cvtColor(roi, rgb, Imgproc.COLOR_RGBA2RGB)
            Imgproc.cvtColor(rgb, lab, Imgproc.COLOR_RGB2Lab)
            val m = Core.mean(lab)
            releaseAll(roi, rgb, lab)
            m
        }
        return Scalar(means.map { it.`val`[0] }.average(), means.map { it.`val`[1] }.average(), means.map { it.`val`[2] }.average(), 0.0)
    }

    private fun maxIndex(values: DoubleArray, start: Int, end: Int): Int {
        var best = start.coerceIn(values.indices)
        var bestVal = Double.NEGATIVE_INFINITY
        for (i in start.coerceAtLeast(0) until end.coerceAtMost(values.size)) {
            if (values[i] > bestVal) {
                bestVal = values[i]
                best = i
            }
        }
        return best
    }

    private fun unionRect(a: CvRect, b: CvRect): CvRect {
        val x1 = min(a.x, b.x)
        val y1 = min(a.y, b.y)
        val x2 = max(a.x + a.width, b.x + b.width)
        val y2 = max(a.y + a.height, b.y + b.height)
        return CvRect(x1, y1, x2 - x1, y2 - y1)
    }

    private fun rectToSquare(r: CvRect, w: Int, h: Int): CvRect {
        val side = max(r.width, r.height).coerceAtLeast(1)
        val cx = r.x + r.width / 2
        val cy = r.y + r.height / 2
        return clampSquare(CvRect(cx - side / 2, cy - side / 2, side, side), w, h)
    }

    private fun clampSquare(r: CvRect, w: Int, h: Int): CvRect {
        val side = min(min(r.width, r.height).coerceAtLeast(1), min(w, h))
        var x = r.x + (r.width - side) / 2
        var y = r.y + (r.height - side) / 2
        x = x.coerceIn(0, w - side)
        y = y.coerceIn(0, h - side)
        return CvRect(x, y, side, side)
    }

    private fun fullCenterSquare(w: Int, h: Int): CvRect {
        val side = min(w, h)
        return CvRect((w - side) / 2, (h - side) / 2, side, side)
    }

    private fun squareAspectScore(r: CvRect): Double {
        val mx = max(r.width, r.height).toDouble().coerceAtLeast(1.0)
        return (1.0 - abs(r.width - r.height) / mx).coerceIn(0.0, 1.0)
    }

    private fun areaReasonScore(ratio: Double): Double = when {
        ratio < 0.05 -> ratio / 0.05 * 0.4
        ratio <= 0.95 -> 1.0
        else -> 0.75
    }

    private fun releaseAll(vararg mats: Mat) = mats.forEach { it.release() }
}
