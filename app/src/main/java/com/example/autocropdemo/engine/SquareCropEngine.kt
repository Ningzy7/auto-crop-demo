package com.example.autocropdemo.engine

import android.graphics.Bitmap
import android.graphics.Rect
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
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
        val debugInfo: String,
        val selected: Boolean = false
    )

    private data class RawCandidate(val method: String, val rect: CvRect, val score: Double)

    fun detectThree(inputBitmap: Bitmap): List<Candidate> {
        val rgba = Mat()
        Utils.bitmapToMat(inputBitmap, rgba)
        val raw = listOf(detectBestByContour(rgba), detectBestByColorDiff(rgba), detectBestByProjection(rgba))
        val out = buildCandidates(rgba, raw)
        rgba.release()
        return out
    }

    fun detectMultiple(inputBitmap: Bitmap, maxCount: Int = 24): List<Candidate> {
        val rgba = Mat()
        Utils.bitmapToMat(inputBitmap, rgba)
        val raw = mutableListOf<RawCandidate>()
        raw += detectContourCandidates(rgba, maxCount)
        raw += detectColorDiffCandidates(rgba, maxCount)
        raw += detectProjectionGridCandidates(rgba, maxCount)
        val merged = mergeCandidates(raw, rgba.cols(), rgba.rows(), maxCount)
        val out = buildCandidates(rgba, merged)
        rgba.release()
        return out
    }

    private fun buildCandidates(rgba: Mat, raw: List<RawCandidate>): List<Candidate> {
        val w = rgba.cols()
        val h = rgba.rows()
        return raw.mapIndexed { index, item ->
            val safe = adaptiveRefineSquare(rgba, clampSquare(item.rect, w, h))
            val roi = Mat(rgba, safe)
            val bmp = Bitmap.createBitmap(roi.cols(), roi.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(roi, bmp)
            roi.release()
            Candidate(
                bitmap = bmp,
                rect = Rect(safe.x, safe.y, safe.x + safe.width, safe.y + safe.height),
                method = item.method,
                score = item.score,
                debugInfo = "#${index + 1} ${item.method}: ${safe.width}x${safe.height}@(${safe.x},${safe.y}), score=${"%.2f".format(item.score)}"
            )
        }
    }

    private fun detectBestByContour(rgba: Mat): RawCandidate = detectContourCandidates(rgba, 1).firstOrNull()
        ?: RawCandidate("轮廓方形", fullCenterSquare(rgba.cols(), rgba.rows()), 0.25)

    private fun detectBestByColorDiff(rgba: Mat): RawCandidate = detectColorDiffCandidates(rgba, 1).firstOrNull()
        ?: RawCandidate("色差方形", fullCenterSquare(rgba.cols(), rgba.rows()), 0.25)

    private fun detectBestByProjection(rgba: Mat): RawCandidate = detectProjectionGridCandidates(rgba, 1).firstOrNull()
        ?: RawCandidate("投影方形", fullCenterSquare(rgba.cols(), rgba.rows()), 0.25)

    private fun detectContourCandidates(rgba: Mat, maxCount: Int): List<RawCandidate> {
        val w = rgba.cols()
        val h = rgba.rows()
        val imageArea = w * h.toDouble()
        val gray = Mat()
        Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.GaussianBlur(gray, gray, Size(5.0, 5.0), 0.0)

        val all = mutableListOf<RawCandidate>()
        val thresholds = listOf(28.0 to 90.0, 40.0 to 120.0, 70.0 to 180.0)
        for ((low, high) in thresholds) {
            val edges = Mat()
            Imgproc.Canny(gray, edges, low, high)
            val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(5.0, 5.0))
            Imgproc.morphologyEx(edges, edges, Imgproc.MORPH_CLOSE, kernel)
            val contours = mutableListOf<MatOfPoint>()
            val hierarchy = Mat()
            Imgproc.findContours(edges.clone(), contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
            for (c in contours) {
                val r0 = Imgproc.boundingRect(c)
                val sq = rectToSquare(expandRect(r0, 0.015, w, h), w, h)
                val areaRatio = (sq.width * sq.height).toDouble() / imageArea
                if (areaRatio in 0.015..0.98) {
                    val peri = Imgproc.arcLength(MatOfPoint2f(*c.toArray()), true)
                    val approx = MatOfPoint2f()
                    Imgproc.approxPolyDP(MatOfPoint2f(*c.toArray()), approx, 0.035 * peri, true)
                    val polyBonus = if (approx.total().toInt() in 4..10) 0.18 else 0.0
                    val score = (squareAspectScore(r0) * 0.45 + areaReasonScore(areaRatio) * 0.25 + edgeStrengthScore(edges, sq) * 0.25 + polyBonus).coerceIn(0.0, 1.2)
                    all += RawCandidate("轮廓", sq, score)
                    approx.release()
                }
                c.release()
            }
            releaseAll(edges, kernel, hierarchy)
        }
        gray.release()
        return mergeCandidates(all, w, h, maxCount)
    }

    private fun detectColorDiffCandidates(rgba: Mat, maxCount: Int): List<RawCandidate> {
        val w = rgba.cols()
        val h = rgba.rows()
        val imageArea = w * h.toDouble()
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

        val all = mutableListOf<RawCandidate>()
        val modes = listOf(Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU, Imgproc.THRESH_BINARY)
        for ((i, mode) in modes.withIndex()) {
            val mask = Mat()
            val threshold = if (mode == Imgproc.THRESH_BINARY) 22.0 else 0.0
            Imgproc.threshold(diffGray, mask, threshold, 255.0, mode)
            val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(7.0 + i * 4.0, 7.0 + i * 4.0))
            Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, kernel)
            val contours = mutableListOf<MatOfPoint>()
            val hierarchy = Mat()
            Imgproc.findContours(mask.clone(), contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
            for (c in contours) {
                val r0 = Imgproc.boundingRect(c)
                val sq = rectToSquare(expandRect(r0, 0.035, w, h), w, h)
                val areaRatio = (sq.width * sq.height).toDouble() / imageArea
                if (areaRatio in 0.015..0.98) {
                    val score = (0.25 + squareAspectScore(r0) * 0.25 + areaReasonScore(areaRatio) * 0.25 + edgeStrengthScore(mask, sq) * 0.25).coerceIn(0.0, 1.1)
                    all += RawCandidate("色差", sq, score)
                }
                c.release()
            }
            releaseAll(mask, kernel, hierarchy)
        }
        releaseAll(rgb, lab, bg, diff, diffGray, *channels.toTypedArray())
        return mergeCandidates(all, w, h, maxCount)
    }

    private fun detectProjectionGridCandidates(rgba: Mat, maxCount: Int): List<RawCandidate> {
        val w = rgba.cols()
        val h = rgba.rows()
        val imageArea = w * h.toDouble()
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
        val xs = topPeakIndexes(colScores, max(4, (w * 0.04).toInt()), 10)
        val ys = topPeakIndexes(rowScores, max(4, (h * 0.04).toInt()), 10)

        val all = mutableListOf<RawCandidate>()
        for (li in xs.indices) for (ri in li + 1 until xs.size) {
            val left = min(xs[li], xs[ri])
            val right = max(xs[li], xs[ri])
            val width = right - left
            if (width < min(w, h) * 0.08) continue
            for (ti in ys.indices) for (bi in ti + 1 until ys.size) {
                val top = min(ys[ti], ys[bi])
                val bottom = max(ys[ti], ys[bi])
                val height = bottom - top
                if (height < min(w, h) * 0.08) continue
                val aspect = 1.0 - abs(width - height).toDouble() / max(width, height)
                if (aspect < 0.72) continue
                val sq = rectToSquare(CvRect(left, top, width, height), w, h)
                val areaRatio = (sq.width * sq.height).toDouble() / imageArea
                if (areaRatio !in 0.015..0.98) continue
                val score = (0.20 + aspect * 0.35 + areaReasonScore(areaRatio) * 0.20 + edgeStrengthScore(gradX, sq) * 0.12 + edgeStrengthScore(gradY, sq) * 0.12).coerceIn(0.0, 1.1)
                all += RawCandidate("投影", sq, score)
            }
        }
        releaseAll(gray, gradX, gradY)
        return mergeCandidates(all, w, h, maxCount)
    }

    private fun mergeCandidates(input: List<RawCandidate>, w: Int, h: Int, maxCount: Int): List<RawCandidate> {
        val sorted = input.map { it.copy(rect = clampSquare(it.rect, w, h)) }.sortedByDescending { it.score }
        val out = mutableListOf<RawCandidate>()
        for (c in sorted) {
            if (out.any { iou(it.rect, c.rect) > 0.55 }) continue
            out += c
            if (out.size >= maxCount) break
        }
        return out.ifEmpty { listOf(RawCandidate("中心兜底", fullCenterSquare(w, h), 0.2)) }
    }

    private fun edgeStrengthScore(edge: Mat, r: CvRect): Double {
        val x1 = r.x.coerceIn(0, edge.cols() - 1)
        val y1 = r.y.coerceIn(0, edge.rows() - 1)
        val x2 = (r.x + r.width - 1).coerceIn(0, edge.cols() - 1)
        val y2 = (r.y + r.height - 1).coerceIn(0, edge.rows() - 1)
        val top = Core.mean(edge.row(y1)).`val`[0]
        val bottom = Core.mean(edge.row(y2)).`val`[0]
        val left = Core.mean(edge.col(x1)).`val`[0]
        val right = Core.mean(edge.col(x2)).`val`[0]
        return ((top + bottom + left + right) / 4.0 / 255.0).coerceIn(0.0, 1.0)
    }

    private fun topPeakIndexes(values: DoubleArray, minDistance: Int, maxPeaks: Int): List<Int> {
        val order = values.indices.sortedByDescending { values[it] }
        val peaks = mutableListOf<Int>()
        val margin = minDistance / 2
        for (i in order) {
            if (i < margin || i >= values.size - margin) continue
            if (peaks.any { abs(it - i) < minDistance }) continue
            peaks += i
            if (peaks.size >= maxPeaks) break
        }
        return peaks.sorted()
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

    private fun adaptiveRefineSquare(rgba: Mat, seed: CvRect): CvRect {
        val w = rgba.cols()
        val h = rgba.rows()
        val side = seed.width.coerceAtLeast(1)
        val search = max(6, (side * 0.10).toInt())
        val gray = Mat()
        Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.GaussianBlur(gray, gray, Size(3.0, 3.0), 0.0)
        val gx = Mat()
        val gy = Mat()
        Imgproc.Sobel(gray, gx, CvType.CV_16S, 1, 0)
        Imgproc.Sobel(gray, gy, CvType.CV_16S, 0, 1)
        Core.convertScaleAbs(gx, gx)
        Core.convertScaleAbs(gy, gy)

        val left = bestVerticalBoundary(gx, seed.x, seed.y, seed.height, -search, search)
        val right = bestVerticalBoundary(gx, seed.x + seed.width - 1, seed.y, seed.height, -search, search)
        val top = bestHorizontalBoundary(gy, seed.y, seed.x, seed.width, -search, search)
        val bottom = bestHorizontalBoundary(gy, seed.y + seed.height - 1, seed.x, seed.width, -search, search)

        val raw = CvRect(
            min(left, right),
            min(top, bottom),
            max(1, abs(right - left) + 1),
            max(1, abs(bottom - top) + 1)
        )
        val refined = rectToSquare(expandRect(raw, 0.008, w, h), w, h)
        releaseAll(gray, gx, gy)
        return clampSquare(refined, w, h)
    }

    private fun bestVerticalBoundary(edge: Mat, baseX: Int, y: Int, height: Int, fromOffset: Int, toOffset: Int): Int {
        var bestX = baseX.coerceIn(0, edge.cols() - 1)
        var bestScore = Double.NEGATIVE_INFINITY
        val y1 = y.coerceIn(0, edge.rows() - 1)
        val y2 = (y + height - 1).coerceIn(0, edge.rows() - 1)
        for (off in fromOffset..toOffset) {
            val x = (baseX + off).coerceIn(0, edge.cols() - 1)
            val roi = Mat(edge, CvRect(x, y1, 1, max(1, y2 - y1 + 1)))
            val score = Core.mean(roi).`val`[0]
            roi.release()
            if (score > bestScore) {
                bestScore = score
                bestX = x
            }
        }
        return bestX
    }

    private fun bestHorizontalBoundary(edge: Mat, baseY: Int, x: Int, width: Int, fromOffset: Int, toOffset: Int): Int {
        var bestY = baseY.coerceIn(0, edge.rows() - 1)
        var bestScore = Double.NEGATIVE_INFINITY
        val x1 = x.coerceIn(0, edge.cols() - 1)
        val x2 = (x + width - 1).coerceIn(0, edge.cols() - 1)
        for (off in fromOffset..toOffset) {
            val y = (baseY + off).coerceIn(0, edge.rows() - 1)
            val roi = Mat(edge, CvRect(x1, y, max(1, x2 - x1 + 1), 1))
            val score = Core.mean(roi).`val`[0]
            roi.release()
            if (score > bestScore) {
                bestScore = score
                bestY = y
            }
        }
        return bestY
    }

    private fun expandRect(r: CvRect, ratio: Double, w: Int, h: Int): CvRect {
        val pad = (max(r.width, r.height) * ratio).toInt().coerceAtLeast(2)
        val x = (r.x - pad).coerceAtLeast(0)
        val y = (r.y - pad).coerceAtLeast(0)
        val x2 = (r.x + r.width + pad).coerceAtMost(w)
        val y2 = (r.y + r.height + pad).coerceAtMost(h)
        return CvRect(x, y, max(1, x2 - x), max(1, y2 - y))
    }

    private fun iou(a: CvRect, b: CvRect): Double {
        val x1 = max(a.x, b.x)
        val y1 = max(a.y, b.y)
        val x2 = min(a.x + a.width, b.x + b.width)
        val y2 = min(a.y + a.height, b.y + b.height)
        val inter = max(0, x2 - x1) * max(0, y2 - y1)
        val union = a.width * a.height + b.width * b.height - inter
        return if (union <= 0) 0.0 else inter.toDouble() / union
    }

    private fun rectToSquare(r: CvRect, w: Int, h: Int): CvRect {
        val side = max(r.width, r.height).coerceAtLeast(1)
        val cx = r.x + r.width / 2
        val cy = r.y + r.height / 2
        return clampSquare(CvRect(cx - side / 2, cy - side / 2, side, side), w, h)
    }

    private fun clampSquare(r: CvRect, w: Int, h: Int): CvRect {
        val side = min(max(r.width, r.height).coerceAtLeast(1), min(w, h))
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
        ratio < 0.03 -> ratio / 0.03 * 0.5
        ratio <= 0.92 -> 1.0
        else -> 0.75
    }

    private fun releaseAll(vararg mats: Mat) = mats.forEach { it.release() }
}
