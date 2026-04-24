package com.example.autocropdemo

import android.animation.AnimatorSet
import android.animation.ObjectAnimator
import android.content.ContentValues
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.graphics.Typeface
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.view.Gravity
import android.view.View
import android.widget.Button
import android.widget.CheckBox
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.setPadding
import androidx.lifecycle.lifecycleScope
import com.example.autocropdemo.engine.SquareCropEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.android.OpenCVLoader
import java.io.OutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class MainActivity : AppCompatActivity() {

    private lateinit var ivOriginal: ImageView
    private lateinit var tvStatus: TextView
    private lateinit var resultContainer: LinearLayout
    private lateinit var heartBurst: TextView

    private val engine = SquareCropEngine()
    private var sourceBitmap: Bitmap? = null
    private var sourceUriString: String? = null
    private var candidates: List<SquareCropEngine.Candidate> = emptyList()
    private val selectedIndexes = linkedSetOf<Int>()

    private val prefs by lazy { getSharedPreferences("session", MODE_PRIVATE) }

    private val pickImage = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri == null) return@registerForActivityResult
        takePersistableReadPermission(uri)
        val bitmap = decodeBitmap(uri)
        if (bitmap == null) {
            toast("加载图片失败")
            return@registerForActivityResult
        }
        sourceUriString = uri.toString()
        prefs.edit()
            .putString(KEY_SOURCE_URI, sourceUriString)
            .putString(KEY_MODE, "")
            .putString(KEY_SELECTED, "")
            .apply()

        sourceBitmap = bitmap
        candidates = emptyList()
        selectedIndexes.clear()
        ivOriginal.setImageBitmap(bitmap)
        resultContainer.removeAllViews()
        tvStatus.text = "已加载：${bitmap.width}x${bitmap.height}，可单目标或多目标识别"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        ivOriginal = findViewById(R.id.ivOriginal)
        tvStatus = findViewById(R.id.tvStatus)
        resultContainer = findViewById(R.id.resultContainer)
        heartBurst = findViewById(R.id.heartBurst)

        val ok = OpenCVLoader.initDebug()
        tvStatus.text = if (ok) "OpenCV 已就绪，先选择图片" else "OpenCV 初始化失败"
        if (!ok) toast("OpenCV 初始化失败")

        findViewById<Button>(R.id.btnPick).setOnClickListener { pickImage.launch(arrayOf("image/*")) }

        findViewById<Button>(R.id.btnAutoCrop).setOnClickListener {
            prefs.edit().putString(KEY_MODE, MODE_THREE).putString(KEY_SELECTED, "").apply()
            runDetection("三路识别中：轮廓 / 色差 / 投影...", clearSelection = true) { bitmap ->
                engine.detectThree(bitmap)
            }
        }

        findViewById<Button>(R.id.btnMultiCrop).setOnClickListener {
            prefs.edit().putString(KEY_MODE, MODE_MULTI).putString(KEY_SELECTED, "").apply()
            runDetection("多目标识别中：生成多个裁切候选...", clearSelection = true) { bitmap ->
                engine.detectMultiple(bitmap, 12)
            }
        }

        findViewById<Button>(R.id.btnSave).setOnClickListener {
            if (selectedIndexes.isEmpty()) {
                toast("请先勾选一个或多个候选图")
                return@setOnClickListener
            }
            lifecycleScope.launch {
                val saved = withContext(Dispatchers.IO) {
                    selectedIndexes.mapNotNull { idx -> savePngToGallery(candidates[idx].bitmap, idx + 1) }.size
                }
                toast("已保存 $saved 张")
                tvStatus.text = "批量保存完成：$saved/${selectedIndexes.size} 张"
                if (saved > 0) playHeartEffect()
            }
        }

        restoreSession(savedInstanceState)
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        outState.putString(KEY_SOURCE_URI, sourceUriString)
        outState.putString(KEY_SELECTED, selectedIndexes.joinToString(","))
    }

    override fun onPause() {
        super.onPause()
        prefs.edit()
            .putString(KEY_SOURCE_URI, sourceUriString)
            .putString(KEY_SELECTED, selectedIndexes.joinToString(","))
            .apply()
    }

    private fun restoreSession(savedInstanceState: Bundle?) {
        val uriText = savedInstanceState?.getString(KEY_SOURCE_URI) ?: prefs.getString(KEY_SOURCE_URI, null)
        if (uriText.isNullOrBlank()) return

        val uri = Uri.parse(uriText)
        val bitmap = decodeBitmap(uri) ?: return
        sourceUriString = uriText
        sourceBitmap = bitmap
        ivOriginal.setImageBitmap(bitmap)

        selectedIndexes.clear()
        val selectedText = savedInstanceState?.getString(KEY_SELECTED) ?: prefs.getString(KEY_SELECTED, "") ?: ""
        selectedText.split(",")
            .mapNotNull { it.toIntOrNull() }
            .forEach { selectedIndexes.add(it) }

        when (prefs.getString(KEY_MODE, "")) {
            MODE_THREE -> runDetection("正在恢复三路候选...", clearSelection = false) { engine.detectThree(it) }
            MODE_MULTI -> runDetection("正在恢复多目标候选...", clearSelection = false) { engine.detectMultiple(it, 12) }
            else -> tvStatus.text = "已恢复上次图片：${bitmap.width}x${bitmap.height}"
        }
    }

    private fun takePersistableReadPermission(uri: Uri) {
        try {
            contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
        } catch (_: Exception) {
            // 部分图片来源不支持持久授权；失败不影响当前会话使用。
        }
    }

    private fun runDetection(
        status: String,
        clearSelection: Boolean = true,
        block: (Bitmap) -> List<SquareCropEngine.Candidate>
    ) {
        val src = sourceBitmap
        if (src == null) {
            toast("请先选择图片")
            return
        }
        tvStatus.text = status
        resultContainer.removeAllViews()
        if (clearSelection) selectedIndexes.clear()
        lifecycleScope.launch {
            val list = withContext(Dispatchers.Default) { block(src) }
            candidates = list
            selectedIndexes.removeAll { it !in list.indices }
            renderCandidates(list)
            tvStatus.text = "已生成 ${list.size} 个候选，勾选后可批量保存 💖"
        }
    }

    private fun renderCandidates(list: List<SquareCropEngine.Candidate>) {
        resultContainer.removeAllViews()
        list.forEachIndexed { index, candidate ->
            val row = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                setPadding(dp(8))
                background = makeCardBg(false)
                val lp = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)
                lp.setMargins(dp(4), dp(5), dp(4), dp(5))
                layoutParams = lp
                isClickable = true
                isFocusable = true
            }

            val check = CheckBox(this).apply {
                isChecked = selectedIndexes.contains(index)
                layoutParams = LinearLayout.LayoutParams(dp(46), LinearLayout.LayoutParams.WRAP_CONTENT)
            }

            val image = ImageView(this).apply {
                setImageBitmap(candidate.bitmap)
                scaleType = ImageView.ScaleType.CENTER_CROP
                layoutParams = LinearLayout.LayoutParams(dp(112), dp(112))
            }

            val textBox = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                setPadding(dp(10), 0, 0, 0)
                layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
            }
            val title = TextView(this).apply {
                text = "${index + 1}. ${candidate.method}"
                typeface = Typeface.DEFAULT_BOLD
                textSize = 15f
            }
            val info = TextView(this).apply {
                text = "score ${"%.2f".format(candidate.score)}\n${candidate.rect.width()}×${candidate.rect.height()} @ (${candidate.rect.left}, ${candidate.rect.top})"
                textSize = 12f
            }
            textBox.addView(title)
            textBox.addView(info)

            fun toggle() {
                if (selectedIndexes.contains(index)) selectedIndexes.remove(index) else selectedIndexes.add(index)
                prefs.edit().putString(KEY_SELECTED, selectedIndexes.joinToString(",")).apply()
                refreshSelectionUi()
                if (selectedIndexes.contains(index)) playHeartEffect()
            }
            row.setOnClickListener { toggle() }
            check.setOnClickListener { toggle() }
            image.setOnClickListener { toggle() }

            row.addView(check)
            row.addView(image)
            row.addView(textBox)
            resultContainer.addView(row)
        }
        refreshSelectionUi()
    }

    private fun refreshSelectionUi() {
        for (i in 0 until resultContainer.childCount) {
            val row = resultContainer.getChildAt(i) as LinearLayout
            row.background = makeCardBg(selectedIndexes.contains(i))
            val cb = row.getChildAt(0) as CheckBox
            cb.isChecked = selectedIndexes.contains(i)
        }
        val selectedText = if (selectedIndexes.isEmpty()) "未选择" else "已选 ${selectedIndexes.size} 张"
        if (candidates.isNotEmpty()) tvStatus.text = "$selectedText，可继续勾选或批量保存"
    }

    private fun makeCardBg(selected: Boolean) = android.graphics.drawable.GradientDrawable().apply {
        cornerRadius = dp(16).toFloat()
        setColor(if (selected) 0xFFFFF0F6.toInt() else 0xFFFFFFFF.toInt())
        setStroke(dp(if (selected) 3 else 1), if (selected) 0xFFFF4081.toInt() else 0xFFE0E0E0.toInt())
    }

    private fun playHeartEffect() {
        heartBurst.visibility = View.VISIBLE
        heartBurst.alpha = 0f
        heartBurst.scaleX = 0.35f
        heartBurst.scaleY = 0.35f
        val set = AnimatorSet()
        set.playTogether(
            ObjectAnimator.ofFloat(heartBurst, View.ALPHA, 0f, 1f, 0f),
            ObjectAnimator.ofFloat(heartBurst, View.SCALE_X, 0.35f, 1.35f),
            ObjectAnimator.ofFloat(heartBurst, View.SCALE_Y, 0.35f, 1.35f),
            ObjectAnimator.ofFloat(heartBurst, View.TRANSLATION_Y, 40f, -80f)
        )
        set.duration = 760
        set.start()
    }

    private fun decodeBitmap(uri: Uri): Bitmap? = try {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            val source = ImageDecoder.createSource(contentResolver, uri)
            ImageDecoder.decodeBitmap(source) { decoder, _, _ -> decoder.isMutableRequired = true }
        } else {
            @Suppress("DEPRECATION") MediaStore.Images.Media.getBitmap(contentResolver, uri)
        }
    } catch (_: Exception) { null }

    private fun savePngToGallery(bitmap: Bitmap, index: Int): Uri? {
        val fileName = "square_crop_${timestamp()}_${index}.png"
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, fileName)
            put(MediaStore.Images.Media.MIME_TYPE, "image/png")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/AutoCropDemo")
                put(MediaStore.Images.Media.IS_PENDING, 1)
            }
        }
        val itemUri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: return null
        return try {
            val out: OutputStream = contentResolver.openOutputStream(itemUri) ?: return null
            out.use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                values.clear()
                values.put(MediaStore.Images.Media.IS_PENDING, 0)
                contentResolver.update(itemUri, values, null, null)
            }
            itemUri
        } catch (_: Exception) {
            contentResolver.delete(itemUri, null, null)
            null
        }
    }

    private fun timestamp(): String = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
    private fun toast(msg: String) = Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()
    private fun dp(v: Int): Int = (v * resources.displayMetrics.density).toInt()

    companion object {
        private const val KEY_SOURCE_URI = "source_uri"
        private const val KEY_SELECTED = "selected_indexes"
        private const val KEY_MODE = "last_mode"
        private const val MODE_THREE = "three"
        private const val MODE_MULTI = "multi"
    }
}
