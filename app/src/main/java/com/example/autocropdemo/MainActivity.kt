package com.example.autocropdemo

import android.animation.AnimatorSet
import android.animation.ObjectAnimator
import android.content.ContentValues
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
    private var candidates: List<SquareCropEngine.Candidate> = emptyList()
    private var selectedIndex: Int = -1

    private val pickImage = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri == null) return@registerForActivityResult
        val bitmap = decodeBitmap(uri)
        if (bitmap == null) {
            toast("加载图片失败")
            return@registerForActivityResult
        }
        sourceBitmap = bitmap
        candidates = emptyList()
        selectedIndex = -1
        ivOriginal.setImageBitmap(bitmap)
        resultContainer.removeAllViews()
        tvStatus.text = "已加载：${bitmap.width}x${bitmap.height}，点击识别方形"
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

        findViewById<Button>(R.id.btnPick).setOnClickListener { pickImage.launch("image/*") }

        findViewById<Button>(R.id.btnAutoCrop).setOnClickListener {
            val src = sourceBitmap
            if (src == null) {
                toast("请先选择图片")
                return@setOnClickListener
            }
            tvStatus.text = "三路识别中：轮廓 / 色差 / 投影..."
            resultContainer.removeAllViews()
            lifecycleScope.launch {
                val list = withContext(Dispatchers.Default) { engine.detectThree(src) }
                candidates = list
                selectedIndex = -1
                renderCandidates(list)
                tvStatus.text = "已生成 ${list.size} 个候选，请点选你喜欢的方形裁切 💖"
            }
        }

        findViewById<Button>(R.id.btnSave).setOnClickListener {
            if (selectedIndex !in candidates.indices) {
                toast("请先点选一个候选图")
                return@setOnClickListener
            }
            val uri = savePngToGallery(candidates[selectedIndex].bitmap)
            if (uri != null) {
                toast("已保存到相册")
                tvStatus.text = "保存成功：${candidates[selectedIndex].debugInfo}"
            } else toast("保存失败")
        }
    }

    private fun renderCandidates(list: List<SquareCropEngine.Candidate>) {
        resultContainer.removeAllViews()
        list.forEachIndexed { index, candidate ->
            val card = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                gravity = Gravity.CENTER
                setPadding(dp(8))
                background = makeCardBg(false)
                val lp = LinearLayout.LayoutParams(dp(190), LinearLayout.LayoutParams.WRAP_CONTENT)
                lp.setMargins(dp(6), dp(6), dp(6), dp(6))
                layoutParams = lp
                isClickable = true
                isFocusable = true
            }

            val title = TextView(this).apply {
                text = candidate.method
                gravity = Gravity.CENTER
                typeface = Typeface.DEFAULT_BOLD
                textSize = 15f
            }
            val image = ImageView(this).apply {
                setImageBitmap(candidate.bitmap)
                scaleType = ImageView.ScaleType.FIT_CENTER
                adjustViewBounds = true
                layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dp(160))
            }
            val info = TextView(this).apply {
                text = "score ${"%.2f".format(candidate.score)}\n${candidate.rect.width()}×${candidate.rect.height()}"
                gravity = Gravity.CENTER
                textSize = 12f
            }
            card.addView(title)
            card.addView(image)
            card.addView(info)
            card.setOnClickListener { selectCandidate(index) }
            resultContainer.addView(card)
        }
    }

    private fun selectCandidate(index: Int) {
        selectedIndex = index
        for (i in 0 until resultContainer.childCount) {
            resultContainer.getChildAt(i).background = makeCardBg(i == index)
        }
        tvStatus.text = "已选择：${candidates[index].debugInfo}"
        playHeartEffect()
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

    private fun savePngToGallery(bitmap: Bitmap): Uri? {
        val fileName = "square_crop_${timestamp()}.png"
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
}
